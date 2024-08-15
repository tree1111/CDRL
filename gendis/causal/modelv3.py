import normflows as nf
import pytorch_lightning as pl
import torch
import torch.optim as optim

from gendis.normalizing_flow.distribution import MultidistrCausalFlow


class CausalNormalizingFlow(nf.NormalizingFlow):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    q0: MultidistrCausalFlow

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of list of flows for each level
          p: Target distribution
        """
        super().__init__(q0=q0, flows=flows, p=p)

    def forward_kld(self, x, intervention_targets=None, distr_idx=None, hard_interventions=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(
            z,
            intervention_targets=intervention_targets,
            e=distr_idx,
            hard_interventions=hard_interventions,
        )
        return -torch.mean(log_q)

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, intervention_targets=None, distr_idx=None, hard_interventions=None):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, intervention_targets, distr_idx, hard_interventions)
        return log_q


class CausalFlowModel(pl.LightningModule):
    model: CausalNormalizingFlow

    def __init__(self, model, lr=1e-3, lr_min=1e-8, lr_scheduler=None):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = model

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

    @torch.no_grad()
    def sample(self, num_samples=1, **params):
        """
        Sample a batch of images from the flow.
        """
        return self.model.sample(num_samples=num_samples, **params)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
            # lr_scheduler_config = {
            #     "scheduler": scheduler,
            #     "interval": "epoch",
            # }
            # config_dict["lr_scheduler"] = lr_scheduler_config
        else:
            # An scheduler is optional, but can help in flows to get the last bpd improvement
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        meta_labels = batch[1]
        distr_idx = meta_labels[:, -1]
        hard_interventions = None
        targets = batch[2]

        # print("Inside training step: ", samples.shape, meta_labels.shape, targets.shape)
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.model.forward_kld(
            samples,
            intervention_targets=targets,
            distr_idx=distr_idx,
            hard_interventions=hard_interventions,
        )
        self.log("train_kld", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # first element of batch is the image tensor
        # second element of batch is the label tensor
        #     "width",
        #     "color",
        #     "fracture_thickness",
        #     "fracture_num_fractures",
        #     "label",
        #     "distr_indicators",
        # third element of batch is the intervention target
        samples = batch[0]
        meta_labels = batch[1]
        distr_idx = meta_labels[:, -1]
        hard_interventions = None
        targets = batch[2]
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.model.forward_kld(
            samples,
            intervention_targets=targets,
            distr_idx=distr_idx,
            hard_interventions=hard_interventions,
        )

        self.log("val_kld", loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     # Perform importance sampling during testing => estimate likelihood M times for each image
    #     samples = []
    #     for _ in range(self.import_samples):
    #         img_ll = self._get_likelihood(batch[0], return_ll=True)
    #         samples.append(img_ll)
    #     img_ll = torch.stack(samples, dim=-1)

    #     # To average the probabilities, we need to go from log-space to exp, and back to log.
    #     # Logsumexp provides us a stable implementation for this
    #     img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

    #     # Calculate final bpd
    #     bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
    #     bpd = bpd.mean()

    #     self.log("test_bpd", bpd)
    #     return bpd
