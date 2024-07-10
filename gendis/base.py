import torch
import torch.nn as nn


class MultienvMultiscaleFlow(nn.Module):
    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """Constructor

        Args:

          q0: Causal base distribution that has latent factors related via a causal graph.
          flows: List of list of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        self.q0 = q0
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.num_levels = len(self.merges)
        self.transform = transform
        self.class_cond = class_cond

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

        final_z = torch.zeros_like(z)

        # iterate through each layer
        for i in range(self.num_levels - 1, -1, -1):
            # within each layer, iterate through each flow
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det

            # apply merge operations
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            final_z[:,]
        self.q0.log_prob(z_, y)
        return log_q

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
