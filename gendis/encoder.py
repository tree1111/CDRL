import numpy as np
import torch
import torch.nn as nn

from gendis.normalizing_flow.distribution import ClusteredCausalDistribution
from normflows.distributions import BaseDistribution


class CausalMultiscaleFlow(nn.Module):
    causalq0: ClusteredCausalDistribution
    noiseq0: BaseDistribution

    def __init__(self, causalq0, flows, merges, transform=None, noiseq0=None, class_cond=True):
        """A causal multiscale flow based model.

        This differs from a standard multiscale flow model in that the base distribution is a causal
        distribution that has latent factors related via a causal graph. The current method
        flattens the latent factors at the final level and uses a causal base distribution
        to model the latent factors.

        Args:

          causalq0: Causal base distribution that has latent factors related via a causal graph.
          # q0: List of base distribution
          flows: List of list of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          noiseq0: (Optional) Noise distribution for latent factors
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        # self.q0 = nn.ModuleList(q0)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.num_levels = len(self.flows)

        # if len(self.q0) != self.num_levels:
        #     raise ValueError("Number of base distributions must match number of levels")
        if len(self.merges) != self.num_levels - 1:
            raise ValueError("Number of flow layers must match number of levels")

        self.transform = transform
        self.class_cond = class_cond
        self.causalq0 = causalq0
        self.noiseq0 = noiseq0
        self.flatten_layer = nn.Flatten()

    def forward(self, x, y=None, reverse=False, inds=None, levels=None):
        """Forward pass

        Args:

          x: Input tensor
          y: (Optional) class label
          reverse: Flag, if True run the model in reverse
          inds: (Optional) indices of the latent factors to return
          levels: (Optional) levels of the latent factors to return
        """
        return -self.log_prob(x, y)

    def log_prob(self, x, y=None, env=None, intervention_targets=None, hard_interventions=None):
        """Get log probability for batch

        Args:
          x: Batch of shape (batch_size, *input_shape)
          y: Classes of x. Must be passed in if `class_cond` is True.

        Returns:
          log probability of shape (batch_size,)
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det

        v_latent = torch.zeros((x.shape[0], np.prod(x.shape[1:])), device=x.device, dtype=x.dtype)
        prev_n_dims = 0

        for i in range(self.num_levels - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det

                # flatten the latent factors that are merged out
                z_ = self.flatten_layer(z_)
                n_dims = z_.shape[1]
                v_latent[:, prev_n_dims : n_dims + prev_n_dims] = z_
                prev_n_dims = n_dims
            # else:
            #     z_ = z
            # if self.class_cond:
            #     log_q += self.q0[i].log_prob(z_, y)
            # else:
            #     log_q += self.q0[i].log_prob(z_)

        # apply a flatten layer
        # v_latent = self.flatten_layer(v_latent)

        # now apply causal distribution
        if intervention_targets is None:
            n_cluster_dims = len(self.causalq0.cluster_sizes)
            intervention_targets = torch.zeros(
                (x.shape[0], n_cluster_dims), device=x.device, dtype=x.dtype
            )
            env = torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)

        if self.noiseq0 is not None:
            log_q += self.noiseq0.log_prob(v_latent[:, : self.noiseq0.n_dim])

            # use the rest of the dimensions to fit the causal model
            v_latent = v_latent[:, self.noiseq0.n_dim :]

        # compute the log-probability of the final V_hat over the causal distribution
        log_q += self.causalq0.log_prob(v_latent, env, intervention_targets, hard_interventions)
        return log_q

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x (image input)

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        if self.transform is not None:
            x, log_det_ = self.transform.inverse(x)
            log_det += log_det_
        z = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_

        # apply flatten layer
        z = torch.cat([self.flatten_layer(z_) for z_ in z], dim=1)

        # optionally "invert" the causal distribution

        return z, log_det

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def sample(
        self, num_samples=1, intervention_targets=None, hard_interventions=None, temperature=None
    ):
        """Samples from flow-based approximate distribution

        Args:
            num_samples: Number of samples to draw
            intervention_targets: Intervention targets for causal distribution
            hard_interventions: Hard interventions for causal distribution
            temperature: Temperature parameter for temp annealed sampling

        Returns:
            Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)

        # first sample from the causal distribution
        z_, log_q_ = self.causalq0(num_samples)

        for i in range(self.num_levels):
            # if self.class_cond:
            #     z_, log_q_ = self.q0[i](num_samples, y)
            # else:
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, "temperature"):
                q0.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not " "support temperature annealed sampling"
                )

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)
