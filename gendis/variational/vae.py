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
        encoder_out_dim,
        lr: float = 1e-4,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out_dim = encoder_out_dim

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

        # map encoder output to the stochastic latent space
        self.encode_to_mean = torch.nn.Linear(self.encoder_out_dim, latent_dim)
        self.encode_to_var = torch.nn.Linear(self.encoder_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))


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

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))


    def training_step(self, batch, batch_idx) -> torch.Tensor | torch.Dict[str, torch.Any]:
        # data tensor and meta-data are passed in the batch
        x = batch[0]

        # forward pass of the model and extract relevant variables
        mu, std, z, x_hat = self.forward(x)

        # first, compute the reconstruction loss
        # recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        # recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

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
        print('\n\n\n', x.min(), x.max(), x_hat.min(), x_hat.max())
        print(x.shape, x_hat.shape)
        # recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        # recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
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
