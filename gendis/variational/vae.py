from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim,
        encoder,
        decoder,
        lr: float = 1e-4,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

        # map encoder output to the stochastic latent space
        self.encode_to_mean = torch.nn.Linear(latent_dim, latent_dim)
        self.encode_to_var = torch.nn.Linear(latent_dim, latent_dim)

        self.save_hyperparameters()

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def encode(self, x):
        hidden = self.encoder(x)

        # map to mean and variance of the latent space
        high_dim_mean = self.encode_to_mean(hidden)
        high_dim_var = self.encode_to_var(hidden)

        return high_dim_mean, high_dim_var

    def decode(self, z):
        return self.decoder(z)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        mean, log_var = self.encode(x)
        std = torch.exp(log_var / 2)

        # sample from the distribution using reparametrization trick
        q_z = torch.distributions.Normal(mean, std)
        z = q_z.rsample()

        x_hat = self.decode(z)
        return mean, std, z, x_hat

    def training_step(self, batch, batch_idx) -> torch.Tensor | torch.Dict[str, torch.Any]:
        # data tensor and meta-data are passed in the batch
        x = batch[0]

        # forward pass of the model and extract relevant variables
        mu, std, z, x_hat = self.forward(x)

        # first, compute the reconstruction loss
        # recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

        # next compute the KL divergence of the latent space from a standard normal distribution
        kl = self.kl_divergence(z, mu, std)

        # now compute the negative ELBO lower bound, which we return to the optimizer to help us minimize
        elbo = -1.0 * (recon_loss - kl).mean()

        self.log("train_kl_loss", kl.mean(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_recon_loss", recon_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", elbo, on_step=True, on_epoch=True, prog_bar=True)

        return elbo

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        mu, std, z, x_hat = self.forward(x)
        # recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        kl = self.kl_divergence(z, mu, std)
        elbo = -1.0 * (recon_loss - kl).mean()
        self.log("validation_loss", elbo, on_step=True, on_epoch=True, prog_bar=True)
        return elbo

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        config_dict = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        config_dict["optimizer"] = optimizer

        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
            config_dict["lr_scheduler"] = lr_scheduler_config
        elif self.lr_scheduler is None:
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        return config_dict
