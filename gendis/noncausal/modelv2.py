import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class MultiscaleGlowFlow(pl.LightningModule):
    model: nf.MultiscaleFlow

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
    def sample(self, num_samples=1):
        """
        Sample a batch of images from the flow.
        """
        return self.model.sample(num_samples=num_samples)

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
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.model.forward_kld(batch[0])
        self.log("train_kld", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.forward_kld(batch[0])
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
