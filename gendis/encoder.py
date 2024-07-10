from typing import List, Optional

import networkx as nx
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from networkx import DiGraph
from normflows.core import MultiscaleFlow
from torch import Tensor, abs, det, log

from .normalizing_flow.distribution import MultidistrCausalFlow, NaiveMultiEnvCausalDistribution
from .normalizing_flow.utils import make_spline_flows


class ClusteredCausalEncoder(nf.NormalizingFlow):
    """Encoder for multi-distributional data that occurs over a C-DAG.

    The encoder maps from the observed data ``X`` to the latent space :math:`\hat{V}`.

    The latent space is assumed to have causal structure according to a specified causal
    graph, ``G``. The encoder is trained to maximize the likelihood of
    the data under the causal model.

    ``X`` and :math:`\hat{V}` are assumed to have the same dimensionality.

    The encoder has two main components:
        1. A causal base distribution q0 over the latent space. This encodes the latent
        causal structure.
        2. An unmixing function mapping from the observations to the latent space.

    Parameters
    ----------
    graph : nx.DiGraph
        The DAG over the representation variables.
    cluster_sizes : np.ndarray of shape (n_clusters, 1)
        The size/dimensionality of each cluster.
    intervention_targets_per_distr : Optional[Tensor], optional
        The intervention targets per distribution, by default None, corresponding
        to no interventions and a single distribution. The single distribution is
        assumed to be observational.
    hard_interventions_per_distr : Tensor of shape (n_distributions, n_clusters)
        Whether the intervention target for each cluster-variable is hard (i.e.
        all parents are removed).
    fix_mechanisms : bool, optional
        Whether to fix the mechanisms, by default False.
    flows : List, optional
        List of normalizing flows, by default None.
    merges : List, optional
        List of merge/split operations (forward pass must do merge).

    Attributes
    ----------
    graph : CausalGraph with latent_dim nodes
        Causal graph of the latent variables possibly with bidirected edges.
    cluster_sizes : Tensor, shape (latent_dim)
        The size of each cluster within the graph.
    intervention_targets_per_distr: Tensor, shape (num_distrs, latent_dim)
        Which variables are intervened on in each distribution.
    hard_interventions_per_distr : Tensor of shape (n_distributions, n_clusters)
        Whether the intervention target for each cluster-variable is hard (i.e.
        all parents are removed).
    latent_dim: int
        Dimension of the latent and observed variables. This is the total dimensionality over all variables
        in the graph. This is equivalent to the sum of the cluster sizes over the entire node set.
    unmixing :
        The unmixing function. Defined by subclasses.

    Methods
    -------
    multi_distr_log_prob(x, e, intervention_targets) -> Tensor
        Computes log probability of ``X`` in distribution ``i``.
    forward(x) -> Tensor
        Maps from the observed data ``X`` to the latent space :math:`\hat{V}`.
    """

    q0: MultidistrCausalFlow

    def __init__(
        self,
        q0: MultidistrCausalFlow,
        graph: nx.DiGraph,
        cluster_sizes: Optional[Tensor] = None,
        intervention_targets_per_distr: Optional[Tensor] = None,
        hard_interventions_per_distr: Optional[Tensor] = None,
        fix_mechanisms: bool = False,
        flows: List = None,
        debug: bool = False,
    ) -> None:
        self.graph = graph
        self.cluster_sizes = cluster_sizes
        self.intervention_targets_per_distr = intervention_targets_per_distr
        self.hard_interventions_per_distr = hard_interventions_per_distr
        self.fix_mechanisms = fix_mechanisms
        self.debug = debug

        super().__init__(
            q0=q0,
            flows=flows if flows is not None else [],
            # merges=merges if merges is not None else [],
        )

    @property
    def latent_dim(self):
        return (
            self.graph.number_of_nodes()
            if self.cluster_sizes is None
            else np.sum(self.cluster_sizes)
        )

    def log_prob(self, v_latent: Tensor, e: Tensor, intervention_targets: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.inverse(x)


class ParametricClusteredCausalEncoder(ClusteredCausalEncoder):
    def __init__(
        self,
        q0: MultidistrCausalFlow,
        graph: nx.DiGraph,
        cluster_sizes: Tensor | None = None,
        intervention_targets_per_distr: Tensor | None = None,
        hard_interventions_per_distr: Tensor | None = None,
        fix_mechanisms: bool = False,
        flows: List = None,
        debug: bool = False,
    ) -> None:
        # define the distributions over the latent space of variables
        if intervention_targets_per_distr is None:
            raise RuntimeError(
                "intervention_targets_per_distr must be provided for parametric base distribution"
            )

        super().__init__(
            q0,
            graph,
            cluster_sizes,
            intervention_targets_per_distr,
            hard_interventions_per_distr,
            fix_mechanisms,
            flows,
            debug=debug,
        )
        self._unmixing = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Log probability of the high-dimensional mixture, X.

        Parameters
        ----------
        x : Tensor of shape (n_distributions, n_dims)
            The high-dimensional mixture.
        e : Tensor of shape (n_distributions, 1)
            Indicator of different environments (overloaded to indicate intervention
            and change in domain).
        intervention_targets : Tensor
            The intervention targets for each variable in each distribution.

        Returns
        -------
        log_q : Tensor
            The log probability of the high-dimensional mixture.
        res : dictionary
            Optional result containing the probability terms, determinant terms and
            log probability.
        """
        # encode the mixture to the latent representation
        v_hat = self(x)

        # compute the jacobian term for mapping x to V_hat
        jacobian = torch.autograd.functional.jacobian(self.inverse, x[0, :], create_graph=True)

        # compute the log determinant of the jacobian
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        log_q += log(abs(det(jacobian)))
        determinant_terms = log_q

        # now compute the log probability of the latent representation
        prob_terms = self.q0.log_prob(v_hat, e, intervention_targets)
        log_q += prob_terms
        res = {
            "log_prob": log_q,
            "determinant_terms": determinant_terms,
            "prob_terms": prob_terms,
        }
        if self.debug:
            return log_q, res
        else:
            return log_q

    def inverse(self, x: Tensor) -> Tensor:
        return self._unmixing(x)


class NonparametricClusteredCausalEncoder(ClusteredCausalEncoder):
    def __init__(
        self,
        q0: MultidistrCausalFlow,
        graph: DiGraph,
        cluster_sizes: Tensor | None = None,
        intervention_targets_per_distr: Tensor | None = None,
        hard_interventions_per_distr: Tensor | None = None,
        fix_mechanisms: bool = False,
        flows: List = None,
        n_flows: int = 3,
        n_hidden_dim: int = 128,
        n_layers: int = 3,
        debug: bool = False,
    ) -> None:
        if flows is None:
            # This is the same as the latent_dim property, but it is not defined yet, so we
            # copy the code here.
            latent_dim = graph.number_of_nodes() if cluster_sizes is None else np.sum(cluster_sizes)
            flows = make_spline_flows(
                n_flows, latent_dim, n_hidden_dim, n_layers, permutation=False
            )

        super().__init__(
            q0,
            graph,
            cluster_sizes,
            intervention_targets_per_distr,
            hard_interventions_per_distr,
            fix_mechanisms,
            flows=flows,
            debug=debug,
        )

        self.flatten_layer = nn.Flatten()

    def log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Log probability of the high-dimensional mixture, X.

        Parameters
        ----------
        x : Tensor of shape (n_distributions, n_dims)
            The high-dimensional mixture.
        e : Tensor of shape (n_distributions, 1)
            Indicator of different environments (overloaded to indicate intervention
            and change in domain).
        intervention_targets : Tensor
            The intervention targets for each variable in each distribution.

        Returns
        -------
        log_q : Tensor
            The log probability of the high-dimensional mixture.
        res : dictionary
            Optional result containing the probability terms, determinant terms and
            log probability.
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)

        # encode X to V_hat and compute the log determinant terms along the way
        # using normalizing flows
        v_latent = x
        for i in range(len(self.flows) - 1, -1, -1):
            v_latent, log_det = self.flows[i].inverse(v_latent)
            log_q += log_det
        determinant_terms = log_q

        # compute the log probability of the final V_hat
        # using the base distribution, which adheres to a latent causal graph
        prob_terms = self.q0.log_prob(v_latent, e, intervention_targets)
        log_q += prob_terms

        res = {
            "log_prob": log_q,
            "determinant_terms": determinant_terms,
            "prob_terms": prob_terms,
        }
        if self.debug:
            return log_q, res
        else:
            return log_q

    def forward(self, x: Tensor) -> Tensor:
        return self.inverse(x)


class NaiveClusteredCausalEncoder(ClusteredCausalEncoder):
    """
    Naive encoder for multi-environment data.

    This encoder does not assume any causal structure in the latent space. Equivalent to independent
    components analysis (ICA).
    """

    def __init__(
        self,
        graph: np.ndarray,
        cluster_sizes: Tensor | None = None,
        intervention_targets_per_distr: Optional[Tensor] = None,
        hard_interventions_per_distr: Tensor | None = None,
        fix_mechanisms: bool = False,
        n_flows: List = 3,
        n_hidden_dim: int = 128,
        n_layers: int = 3,
        debug: bool = False,
    ) -> None:
        # q0
        q0 = NaiveMultiEnvCausalDistribution(
            adjacency_matrix=graph,
        )

        super().__init__(
            graph=graph,
            cluster_sizes=cluster_sizes,
            intervention_targets_per_distr=intervention_targets_per_distr,
            hard_interventions_per_distr=hard_interventions_per_distr,
            fix_mechanisms=fix_mechanisms,
            n_flows=n_flows,
            n_hidden_dim=n_hidden_dim,
            n_layers=n_layers,
            q0=q0,
            debug=debug,
        )

    def log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        v_latent = x
        for i in range(len(self.flows) - 1, -1, -1):
            v_latent, log_det = self.flows[i].inverse(v_latent)
            log_q += log_det
        determinant_terms = log_q
        prob_terms = self.q0.log_prob(v_latent, e, intervention_targets)
        log_q += prob_terms

        res = {
            "log_prob": log_q,
            "determinant_terms": determinant_terms,
            "prob_terms": prob_terms,
        }
        if self.debug:
            return log_q, res
        else:
            return log_q

    def forward(self, x: Tensor) -> Tensor:
        return self.inverse(x)
