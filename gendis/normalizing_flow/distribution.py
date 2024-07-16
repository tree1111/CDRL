from abc import ABC

import networkx as nx
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from normflows.distributions import DiagGaussian
from torch import Tensor
from torch.nn.functional import gaussian_nll_loss

from .utils import make_spline_flows, set_initial_edge_coeffs, set_initial_noise_parameters


class MultidistrCausalFlow(nf.distributions.BaseDistribution, ABC):
    """
    Base class for parametric multi-environment causal distributions.

    In typical normalizing flow architectures, the base distribution is a simple distribution
    such as a multivariate Gaussian. In our case, the base distribution has additional multi-environment
    causal structure. Hence, in the parametric case, this class learns the parameters of the causal
    mechanisms and noise distributions. The causal graph is assumed to be known.

    This is a subclass of BaseDistribution, which is a subclass of torch.nn.Module. Hence, this class
    can be used as a base distribution in a normalizing flow.

    Methods
    -------
    log_prob(z, e, intervention_targets) -> Tensor
        Compute the log probability of the latent variables v in environment e, given the intervention targets.
        This is used as the main training objective.
    """

    def log_prob(self, z: Tensor, e: Tensor, intervention_targets: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, num_samples=1, intervention_targets: Tensor = None):
        raise NotImplementedError


class ClusteredCausalDistribution(MultidistrCausalFlow):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        cluster_sizes: np.ndarray,
        intervention_targets_per_distr: Tensor,
        hard_interventions_per_distr: Tensor,
        fix_mechanisms: bool = False,
        use_matrix: bool = False,
    ):
        """Parametric distribution over a clustered causal graph.

        A clustered causal graph defines a grouping over the variables in the graph.
        Thus the variables that describe the graph can either be the clusters, or the
        fine-grained variables.

        Parameters
        ----------
        adjacency_matrix : np.ndarray of shape (n_clusters, n_clusters)
            The adjacency matrix of the causal graph over the clustered variables.
            Each row/column is another cluster.
        cluster_sizes : np.ndarray of shape (n_clusters, 1)
            The size/dimensionality of each cluster.
        intervention_targets_per_distr : Tensor of shape (n_distributions, n_clusters)
            The intervention targets for each cluster-variable in each environment.
        hard_interventions_per_distr : Tensor of shape (n_distributions, n_clusters)
            Whether the intervention target for each cluster-variable is hard (i.e.
            all parents are removed).
        fix_mechanisms : bool, optional
            Whether to fix the mechanisms, by default False.
        use_matrix : bool, optional
            Whether to use a matrix to represent the edge coefficients, by default False.
            If False, a vector is used and the kronecker product is used to compute the
            product of the coefficients with the parent variables.

        Attributes
        ----------
        coeff_values : nn.ParameterList of length (n_nodes) each of length (cluster_dim + 1)
            The coefficients for the linear mechanisms for each variable in the DAG.
            The last element in the list is the constant term.
        noise_means : nn.ParameterList of length (n_nodes) each of length (cluster_dim)
            The means for the noise distributions for each variable in the DAG.
        noise_stds : nn.ParameterList of length (n_nodes) each of length (cluster_dim)
            The standard deviations for the noise distributions for each variable in the DAG.
        """
        super().__init__()

        self.adjacency_matrix = adjacency_matrix
        self.intervention_targets_per_distr = intervention_targets_per_distr
        self.hard_interventions_per_distr = hard_interventions_per_distr
        self.use_matrix = use_matrix

        if cluster_sizes is None:
            cluster_sizes = [1] * adjacency_matrix.shape[0]
        self.cluster_sizes = cluster_sizes

        # map the node in adjacency matrix to a cluster size in the latent space
        self.cluster_mapping = dict()
        for idx in range(len(cluster_sizes)):
            start = np.sum(cluster_sizes[:idx])
            end = start + cluster_sizes[idx]
            self.cluster_mapping[idx] = (start, end)

        self.dag = nx.DiGraph(adjacency_matrix)
        self.latent_dim = (
            self.dag.number_of_nodes() if self.cluster_sizes is None else np.sum(self.cluster_sizes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # parametrize the trainable coefficients for the linear mechanisms
        # this will be a full matrix of coefficients for each variable in the DAG
        coeff_values, coeff_values_requires_grad = set_initial_edge_coeffs(
            self.dag,
            min_val=-1.0,
            max_val=1.0,
            cluster_mapping=self.cluster_mapping,
            use_matrix=self.use_matrix,
            device=device,
        )
        environments = torch.ones(intervention_targets_per_distr.shape[0], 1, device=device)

        # parametrize the trainable means for the noise distributions
        # for each separate distribution
        noise_means, noise_means_requires_grad = set_initial_noise_parameters(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_distr,
            environments=environments,
            n_dim_per_node=cluster_sizes,
            min_val=-0.5,
            max_val=0.5,
            device=device,
        )

        # parametrize the trainable standard deviations for the noise distributions
        # for each separate distribution
        noise_stds, noise_stds_requires_grad = set_initial_noise_parameters(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_distr,
            n_dim_per_node=cluster_sizes,
            environments=environments,
            min_val=0.5,
            max_val=1.5,
            device=device,
        )

        self.coeff_values = nn.ParameterList(coeff_values)
        self.noise_means = nn.ParameterList(noise_means)
        self.noise_stds = nn.ParameterList(noise_stds)
        self.coeff_values_requires_grad = coeff_values_requires_grad
        self.noise_means_requires_grad = noise_means_requires_grad
        self.noise_stds_requires_grad = noise_stds_requires_grad

    def forward(
        self, num_samples=1, intervention_targets: Tensor = None, hard_interventions: Tensor = None
    ):
        if intervention_targets is not None:
            if intervention_targets.squeeze().shape[0] != self.dag.number_of_nodes():
                raise ValueError(
                    "Intervention targets must have the same length as the number of nodes in the DAG."
                )
            if hard_interventions is not None and len(intervention_targets) != len(
                hard_interventions
            ):
                raise ValueError(
                    "Intervention targets and hard interventions must have the same length."
                )
        if hard_interventions is None:
            hard_interventions = torch.zeros(self.latent_dim)
        if intervention_targets is None:
            intervention_targets = torch.zeros_like(hard_interventions)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # return (num_samples, latent_dim) samples from the distribution
        samples = torch.zeros((num_samples, self.latent_dim), device=device)

        # return the log probability of the samples
        log_p = torch.zeros(num_samples, device=device)

        # start from observational environment
        noise_env_idx = 0

        # sample from the noise distributions for each variable over the DAG
        for idx in range(self.dag.number_of_nodes()):
            parents = list(self.dag.predecessors(idx))

            # get start/end in the representation for this cluster
            start, end = self.cluster_mapping[idx]
            cluster_idx = np.arange(start, end, dtype=int)

            # compute the contribution of the parents
            if len(parents) == 0 or (
                intervention_targets[idx] == 1 and hard_interventions[idx] == 1
            ):
                parent_contribution = 0.0
                var = self.noise_stds[noise_env_idx][idx] ** 2
            else:
                parent_cluster_idx = np.hstack(
                    [np.arange(*self.cluster_mapping[p], dtype=int) for p in parents]
                )
                # get coeffieicnts for the parents
                # which is a vector of coefficients for each parent
                coeffs_raw = self.coeff_values[idx][:-1]
                if isinstance(coeffs_raw, nn.ParameterList):
                    coeffs_raw = torch.cat([c for c in coeffs_raw])
                parent_coeffs = coeffs_raw.to(device)
                parent_contribution = parent_coeffs * samples[:, parent_cluster_idx]

                # compute the contribution of the noise
                var = self.noise_stds[noise_env_idx][idx] ** 2 * torch.ones_like(
                    samples[:, cluster_idx]
                )

            # XXX: compute the contributions of the confounders

            noise_coeff = self.coeff_values[idx][-1].to(device)
            noise_contribution = noise_coeff * self.noise_means[noise_env_idx][idx]
            var *= noise_coeff**2
            var *= noise_coeff**2

            # print(parent_contribution, noise_contribution.shape, var.shape, samples[:, idx].shape)

            # samples from the normal distribution for (n_samples, cluster_dims)
            samples[:, cluster_idx] = torch.normal(
                parent_contribution + noise_contribution, var.sqrt()
            )

            # compute the log probability of the variable given the
            # parametrized normal distribution using the parents mean and variance
            log_p += (
                torch.distributions.Normal(parent_contribution + noise_contribution, var.sqrt())
                .log_prob(samples[:, cluster_idx])
                .mean(axis=1)
            )
        return samples, log_p

    def log_prob(
        self,
        v_latent: Tensor,
        e: Tensor,
        intervention_targets: Tensor,
        hard_interventions: Tensor = None,
    ) -> Tensor:
        """Multi-environment log probability of the latent variables.

        Parameters
        ----------
        v_latent : Tensor of shape (n_distributions, latent_dim)
            The "representation" layer for latent variables v.
        e : Tensor of shape (n_distributions, 1)
            Indicator of different environments (overloaded to indicate intervention
            and change in domain).
        intervention_targets : Tensor
            The intervention targets for each variable in each distribution.

        Returns
        -------
        log_p : Tensor of shape (n_distributions, 1)
            The log probability of the latent variables in each distribution.
        """
        log_p = torch.zeros(len(v_latent), dtype=v_latent.dtype, device=v_latent.device)
        latent_dim = v_latent.shape[1]
        if hard_interventions is None:
            hard_interventions = torch.zeros_like(intervention_targets)

        for env in e.unique():
            env_mask = (e == env).flatten()

            v_env = v_latent[env_mask, :]
            intervention_targets_env = intervention_targets[env_mask, :]
            hard_interventions_env = hard_interventions[env_mask, :]

            # iterate over all variables in the latent space in topological order
            for idx in range(self.dag.number_of_nodes()):
                parents = list(self.dag.predecessors(idx))

                # get start/end in the representation for this cluster
                start, end = self.cluster_mapping[idx]
                cluster_idx = np.arange(start, end, dtype=int)

                noise_env_idx = int(env) if intervention_targets_env[0, idx] == 1 else 0

                # compute the contribution of the parents
                if len(parents) == 0 or (
                    intervention_targets_env[0, idx] == 1 and hard_interventions_env[0, idx] == 1
                ):
                    parent_contribution = 0.0
                    var = self.noise_stds[noise_env_idx][idx] ** 2
                else:
                    parent_cluster_idx = np.hstack(
                        [np.arange(*self.cluster_mapping[p], dtype=int) for p in parents]
                    )

                    # get coeffieicnts for the parents
                    # which is a vector of coefficients for each parent
                    coeffs_raw = self.coeff_values[idx][:-1]
                    if isinstance(coeffs_raw, nn.ParameterList):
                        coeffs_raw = torch.cat([c for c in coeffs_raw])

                    parent_coeffs = coeffs_raw.to(v_latent.device)
                    parent_contribution = parent_coeffs * v_env[:, parent_cluster_idx]

                    # compute the contribution of the noise
                    var = self.noise_stds[noise_env_idx][idx] ** 2 * torch.ones_like(
                        v_env[:, cluster_idx]
                    )

                # XXX: compute the contributions of the confounders

                noise_coeff = self.coeff_values[idx][-1].to(v_latent.device)
                noise_contribution = noise_coeff * self.noise_means[noise_env_idx][idx]
                var *= noise_coeff**2

                # compute the log probability of the variable given the
                # parametrized normal distribution using the parents mean and variance
                log_p_distr = (
                    torch.distributions.Normal(parent_contribution + noise_contribution, var.sqrt())
                    .log_prob(v_env[:, cluster_idx])
                    .sum(axis=1)
                )

                log_p[env_mask] += log_p_distr

        return log_p


class NaiveMultiEnvCausalDistribution(MultidistrCausalFlow):
    """
    Naive multi-environment causal distribution.

    This is a dummy-version of ParamMultiEnvCausalDistribution, where the causal mechanisms are assumed to
    be trivial (no connectioons between variables) and the noise distributions are assumed to be Gaussian
    and independent of the environment. This is equivalent to the independent component analysis (ICA) case.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix

        self.q0 = DiagGaussian(adjacency_matrix.shape[0], trainable=True)

    def log_prob(self, z: Tensor, e: Tensor, intervention_targets: Tensor) -> Tensor:
        return self.q0.log_prob(z)


class MultiEnvBaseDistribution(nf.distributions.BaseDistribution):
    """
    Base distribution for nonparametric multi-environment causal distributions.

    This simple independent Gaussian distribution is used as the base distribution for the
    nonparametric multi-environment causal distribution. I.e. this distribution represents the
    exogenous noise in the SCM.
    """

    def __init__(self, shape, trainable=True):
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def log_prob(self, x: Tensor, e: Tensor, intervention_targets: Tensor) -> Tensor:
        # compute the log-likelihood over the non-intervened targets for each sample
        gaussian_nll = gaussian_nll_loss(
            x, torch.zeros_like(x), torch.ones_like(x), full=True, reduction="none"
        )
        mask = ~intervention_targets.to(bool)
        log_p = -(mask * gaussian_nll).sum(dim=1)
        return log_p

    def forward(
        self,
        num_samples=1,
        intervention_targets: Tensor = None,
        hard_interventions: Tensor = None,
        mean_shift=None,
        std_scale=None,
    ):
        # XXX: unsure how to sample when the latent variables are interevened
        # 1. Do we intervene by perturbing the exogenous distribution?
        # 2. Do we intervene by perturbing the latent variables in the NonParametricBaseDistribution?
        #   - it is unclear
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # for intervention, we shift either the mean or scale of the distribution from
        # the standard normal distribution
        mean = 0.0
        std = 1.0
        if mean_shift is not None:
            mean += mean_shift
        if std_scale is not None:
            std *= std_scale
        eps = torch.normal(mean=mean, std=std, size=(num_samples, self.latent_dim), device=device)

        # apply a temperature to the log scaling
        log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.latent_dim + 1))
        )
        return z, log_p


class NonparametricClusteredCausalDistribution(nf.NormalizingFlow):
    """NonParametric distribution over a clustered causal graph.

    A clustered causal graph defines a grouping over the variables in the graph.
    Thus the variables that describe the graph can either be the clusters, or the
    fine-grained variables.

    Parameters
    ----------
    adjacency_matrix : np.ndarray of shape (n_clusters, n_clusters)
        The adjacency matrix of the causal graph over the clustered variables.
        Each row/column is another cluster.
    cluster_sizes : np.ndarray of shape (n_clusters, 1)
        The size/dimensionality of each cluster.
    intervention_targets_per_distr : Tensor of shape (n_distributions, n_clusters)
        The intervention targets for each cluster-variable in each environment.
    hard_interventions_per_distr : Tensor of shape (n_distributions, n_clusters)
        Whether the intervention target for each cluster-variable is hard (i.e.
        all parents are removed).
    fix_mechanisms : bool, optional
        Whether to fix the mechanisms, by default False.
    n_flows : int, optional
        Number of normalizing flows to use, by default 3.
    n_layers : int, optional
        Number of normalizing flow layers to use, by default 3.
    n_hidden_dim : int, optional
        Number of dimensions in the normalizing flow, by default 128.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        cluster_sizes: np.ndarray,
        intervention_targets_per_distr: Tensor,
        hard_interventions_per_distr: Tensor,
        fix_mechanisms: bool = False,
        n_flows: int = 3,
        n_layers: int = 3,
        n_hidden_dim: int = 128,
    ):
        self.adjacency_matrix = adjacency_matrix
        self.cluster_sizes = cluster_sizes
        self.intervention_targets_per_distr = intervention_targets_per_distr
        self.hard_interventions_per_distr = hard_interventions_per_distr
        self.fix_mechanisms = fix_mechanisms

        self.n_flows = n_flows
        self.n_layers = n_layers
        self.n_hidden_dim = n_hidden_dim

        self.dag = nx.DiGraph(adjacency_matrix)
        self.latent_dim = (
            self.dag.number_of_nodes() if self.cluster_sizes is None else np.sum(self.cluster_sizes)
        )

        # get the permutation according to topological order for the variables
        # in the latent space
        self.perm = []
        start = 0
        for idx in list(nx.topological_sort(nx.DiGraph(self.adjacency_matrix))):
            self.perm.extend(range(start, cluster_sizes[idx] + start))
            start = cluster_sizes[idx]
        self.perm = torch.tensor(self.perm)

        flows = make_spline_flows(
            n_flows, self.latent_dim, n_hidden_dim, n_layers, permutation=False
        )
        q0: MultiEnvBaseDistribution = MultiEnvBaseDistribution()

        super().__init__(q0, flows)

    def forward(
        self, num_samples=1, intervention_targets: Tensor = None, hard_interventions: Tensor = None
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # return (num_samples, latent_dim) samples from the distribution
        # samples = torch.zeros((num_samples, self.latent_dim), device=device)

        # we will sample the base distribution
        # samples = self.q0.sample(num_samples, intervention_targets)
        # then we will map the samples to the latent variables

        # XXX: see MultiEnvBaseDistribution
        raise RuntimeError("Not implemented.")

    def log_prob(self, v_latent: Tensor, e: Tensor, intervention_targets: Tensor) -> Tensor:
        """Log probability of the latent variables.

        This method computes the log probability of the latent variables given the
        intervention targets and the environment indicator. It combines the
        determinant terms from the flow transformations with the base distribution
        probabilities and the adjustments for any intervened mechanisms.

        Parameters
        ----------
        v_latent : Tensor of shape (n_distributions, n_nodes)
            The "representation" layer for latent variables v.
        e : Tensor of shape (n_distributions, 1)
            Indicator of different environments (overloaded to indicate intervention
            and change in domain).
        intervention_targets : Tensor
            The intervention targets for each variable in each distribution.

        Returns
        -------
        log_q : Tensor
            The log probability of the latent representation.
        """
        representation_intervention_targets = intervention_target_to_cluster_target(
            intervention_targets, self.cluster_sizes
        )

        # permute inputs to be in topological order
        # do we need the topological order
        v_latent = v_latent[:, self.perm]

        # map latent V to the exogenous variables
        log_q, u = self._determinant_terms(representation_intervention_targets, v_latent)

        # compute the log probability over 'u' for the non-intervened targets
        prob_terms = self.q0.log_prob(u, e, representation_intervention_targets)

        # compute the log probability over 'v_latent' for the intervened targets
        prob_terms_intervened = self._prob_terms_intervened(
            representation_intervention_targets, v_latent
        )
        log_q += prob_terms + prob_terms_intervened

        return log_q

    def _determinant_terms(
        self, intervention_targets: Tensor, v_latent: Tensor
    ) -> tuple[Tensor, Tensor]:
        """_summary_

        Parameters
        ----------
        intervention_targets : Tensor of shape (n_distributions, n_clusters)
            The intervention targets for each cluster-variable in each environment.
        v_latent : Tensor of shape (n_distributions, latent_dim)
            The latent variables after applying the flow transformations. These are the
            input coming from a neural network output, which are mapped to exogenous noise
            variables.

        Returns
        -------
        log_q : Tensor of shape (n_distributions,)
            The log determinant of the Jacobian for the transformations applied to the latent variables.
            This is the :math:`|\\log \\det J_{\widehat{f^{-1}}}|`, which is the Jacobian matrix of the
            mapping from V to u.
        u : Tensor of shape (n_distributions, n_clusters)
            The latent variables before applying the flow transformations.
        """
        log_q = torch.zeros(len(v_latent), dtype=v_latent.dtype, device=v_latent.device)
        u = v_latent
        for i in range(len(self.flows) - 1, -1, -1):
            u, log_det = self.flows[i].inverse(u)
            log_q += log_det

        # remove determinant terms for intervened mechanisms
        jac_row = torch.autograd.functional.jvp(
            self.inverse, v_latent, v=intervention_targets, create_graph=True
        )[1]
        jac_diag_element = (jac_row * intervention_targets).sum(dim=1)

        # mask zero elements
        not_intervened_mask = ~intervention_targets.sum(dim=1).to(bool)
        jac_diag_element[not_intervened_mask] = 1
        log_q -= torch.log(abs(jac_diag_element) + 1e-8)
        return log_q, u

    def _prob_terms_intervened(self, intervention_targets: Tensor, v_latent: Tensor) -> Tensor:
        """
        Compute the probability terms for the intervened mechanisms.

        This method calculates the Gaussian negative log-likelihood (NLL) for the
        intervened variables, adjusting for the fact that these variables have been
        directly manipulated and thus their distributions differ from the original
        causal mechanisms.

        Parameters
        ----------
        intervention_targets : Tensor of shape (n_distributions, n_clusters)
            The intervention targets for each cluster-variable in each environment.
        v_latent : Tensor of shape (n_distributions, n_clusters)
            The "representation" layer for latent variables v.

        Returns
        -------
        prob_terms_intervention_targets : Tensor of shape (n_distributions,)
            The probability terms for the intervened mechanisms, accounting for the
            modifications to the causal graph.
        """
        gaussian_nll = gaussian_nll_loss(
            v_latent,
            torch.zeros_like(v_latent),
            torch.ones_like(v_latent),
            full=True,
            reduction="none",
        )
        mask = intervention_targets.to(bool)
        prob_terms_intervention_targets = -(mask * gaussian_nll).sum(dim=1)
        return prob_terms_intervention_targets


def intervention_target_to_cluster_target(
    intervention_targets: Tensor, cluster_sizes: np.ndarray
) -> Tensor:
    """Convert intervention targets to cluster targets.

    Parameters
    ----------
    intervention_targets : Tensor of shape (n_distributions, n_clusters)
        The intervention targets for each cluster-variable in each environment.
    cluster_sizes : np.ndarray of shape (n_clusters, 1)
        The size/dimensionality of each cluster.

    Returns
    -------
    cluster_targets : Tensor of shape (n_distributions, latent_dim)
        The intervention targets for each variable in each distribution.
    """
    cluster_targets = torch.zeros(
        intervention_targets.shape[0], np.sum(cluster_sizes), device=intervention_targets.device
    )
    start = 0
    for idx, cluster_size in enumerate(cluster_sizes):
        end = start + cluster_size
        cluster_targets[:, start:end] = intervention_targets[:, idx].unsqueeze(1)
        start = end
    return cluster_targets
