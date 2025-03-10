import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import save_image


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class VAE(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        alpha: int,
        batch_size: int,
        lr: float = 1e-3,
        lr_min: float = 1e-7,
        lr_scheduler: Optional[str] = None,
        save_images: bool = False,
        save_path: Optional[str] = None,
    ):
        """Init function for the VAE

        Args:

        hidden_size (int): Latent Hidden Size
        alpha (int): Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        lr (float): Learning Rate, will not be used if auto_lr_find is used.
        save_images (Optional[bool]): Boolean to decide whether to save images
        save_path (Optional[str]): Path to save images
        """

        super().__init__()
        self.hidden_size = hidden_size
        if save_images:
            self.save_path = f"{save_path}"
        self.save_hyperparameters()
        self.save_images = save_images
        self.lr = lr
        self.lr_min = lr_min
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, 392),
            nn.BatchNorm1d(392),
            nn.LeakyReLU(0.1),
            nn.Linear(392, 196),
            nn.BatchNorm1d(196),
            nn.LeakyReLU(0.1),
            nn.Linear(196, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, hidden_size),
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.alpha = alpha
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 196),
            nn.BatchNorm1d(196),
            nn.LeakyReLU(0.1),
            nn.Linear(196, 392),
            nn.BatchNorm1d(392),
            nn.LeakyReLU(0.1),
            nn.Linear(392, 784),
            Stack(1, 28, 28),
            nn.Tanh(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss * self.alpha + kl_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss * self.alpha + kl_loss
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # print(x.mean(),x_out.mean())
        return x_out, x, loss

    def validation_epoch_end(self, outputs):

        if not self.save_images:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        choice = random.choice(outputs)
        xhat_sample = choice[0].reshape(-1, 3, 28, 28)
        x_sample = choice[1].reshape(-1, 3, 28, 28)
        # output_sample = self.scale_image(output_sample)
        save_image(
            xhat_sample[:20],
            f"{self.save_path}/xhat_epoch_{self.current_epoch+1}.png",
            # value_range=(-1, 1)
        )
        save_image(
            x_sample[:20],
            f"{self.save_path}/x_epoch_{self.current_epoch+1}.png",
            # value_range=(-1, 1)
        )

    def configure_optimizers(self):
        config_dict = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        config_dict["optimizer"] = optimizer
        config_dict["monitor"] = "val_loss"

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

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        output = self.decoder(hidden)
        return mu, log_var, output

    def scale_image(self, img):
        out = (img + 1) / 2
        return out

    def interpolate(self, x1, x2):

        assert x1.shape == x2.shape, "Inputs must be of the same shape"
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the "
                "function"
            )
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1.0 - wt) * z1 + wt * z2
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=0).squeeze(1)
        return out, (mu1, lv1), (mu2, lv2)
