from itertools import product
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from .encoder import CausalMultiscaleFlow
from .metrics import mean_correlation_coefficient


def misspecify_adjacency(graph: np.ndarray):
    if graph.shape[0] == 2 and np.sum(graph) == 0:
        # for 2 variables, if adjacency matrix is [[0, 0], [0, 0]], then
        # replace with [[0, 1], [0, 0]]

        graph_out = np.zeros_like(graph)
        graph_out[0, 1] = 1
        return graph_out
    elif graph.shape[0] > 2:
        raise ValueError(
            "Adjacency misspecification not supported for empty adjacency matrix for >2 variables"
        )
    else:
        return graph.T


class NeuralClusteredASCMFlow(pl.LightningModule):
    """Neural Clustered Augmented SCM Flow Model.

    Main class for Neural augmented structural causal models (NASCM-Flow). It implements the
    training loop and the evaluation metrics.

    The model is an encoder-decoder model where the encoding and decoding uses flows as the
    layers (i.e. invertible transformations).

    Attributes
    ----------
    encoder : CausalMultiscaleFlow
        The causal encoder. Needs to be set in subclasses. The inverse of the encoder is the
        unmixing function.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.

    Methods
    -------
    training_step(batch, batch_idx) -> Tensor
        Training step.
    validation_step(batch, batch_idx) -> dict[str, Tensor]
        Validation step: basically passes data to validation_epoch_end.
    validation_epoch_end(outputs) -> None
        Computes validation metrics across all validation data.
    test_step(batch, batch_idx) -> dict[str, Tensor]
        Test step: basically passes data to test_epoch_end.
    test_epoch_end(outputs) -> None
        Computes test metrics across all test data.
    configure_optimizers() -> dict | torch.optim.Optimizer
        Configures the optimizer and learning rate scheduler.
    forward(x) -> torch.Tensor
        Computes the latent variables from the observed data.
    on_before_optimizer_step(optimizer, optimizer_idx) -> None
        Callback that is called before each optimizer step. It ensures that some gradients
        are set to zero to fix some causal mechanisms. See documentation of ParamMultiEnvCausalDistribution
        for more details.
    """

    encoder: CausalMultiscaleFlow  # set in subclasses

    def __init__(
        self,
        encoder,
        lr: float = 1e-4,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

        self.save_hyperparameters()

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """Apply training step over batch of data.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            A batch of data containing the following tensors:
            - x: Tensor of shape (n_samples, n_vars)
                The observed data.
            - v: Tensor of shape (n_samples, n_outputs)
                The ground-truth latent variables.
            - u: Tensor of shape (n_samples, n_outputs)
                The ground-truth interventions.
            - e: Tensor of shape (n_samples, n_envs)
                The environment indicators.
            - int_target: Tensor of shape (n_samples, n_outputs)
                The intervention targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        loss : Tensor
            The average loss over the batch, which is the negative log likelihood.
        """
        (x, meta_labels) = batch
        print(x.shape, len(meta_labels), type(meta_labels))
        width = meta_labels[0]
        color = meta_labels[1]
        fracture_thickness = meta_labels[2]
        fracture_num_fractures = meta_labels[3]
        label = meta_labels[4]
        distr_indicators = meta_labels[5]
        intervention_targets = meta_labels[6]
        log_prob = self.encoder.log_prob(
            x, y=None, env=distr_indicators, intervention_targets=intervention_targets
        )
        loss = -log_prob.mean()
        self.log(f"train_loss", loss, prog_bar=False)

        # compute bits per dimension
        bpd = log_prob * np.log2(np.exp(1)) / np.prod(x.shape[1:])
        bpd = bpd.mean()
        self.log(f"train_bpd", bpd, prog_bar=False)

        return bpd

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> dict[str, Tensor]:
        """Validation step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            A batch of data containing the following tensors:
            - x: Tensor of shape (n_samples, n_vars)
                The observed data.
            - v: Tensor of shape (n_samples, n_outputs)
                The ground-truth latent variables.
            - u: Tensor of shape (n_samples, n_outputs)
                The ground-truth interventions.
            - e: Tensor of shape (n_samples, n_envs)
                The environment indicators.
            - int_target: Tensor of shape (n_samples, n_outputs)
                The intervention targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict[str, Tensor]
            Returns:
                - log_prob: Tensor of shape (n_samples,)
                    The log probability of the data given the encoder.
                - v: Tensor of shape (n_samples, n_outputs)
                    The ground-truth latent variables.
                - v_hat: Tensor of shape (n_samples, n_outputs)
                    The estimated latent variables.
        """
        (x, meta_labels) = batch
        print(x.shape, len(meta_labels), type(meta_labels))
        width = meta_labels[0]
        color = meta_labels[1]
        fracture_thickness = meta_labels[2]
        fracture_num_fractures = meta_labels[3]
        label = meta_labels[4]
        distr_indicators = meta_labels[5]
        intervention_targets = meta_labels[6]
        log_prob = self.encoder.log_prob(
            x, y=None, env=distr_indicators, intervention_targets=intervention_targets
        )

        # compute bits per dimension
        bpd = log_prob * np.log2(np.exp(1)) / np.prod(x.shape[1:])

        v_hat = self(x)
        # print(
        #     log_prob.shape,
        #     x.shape,
        #     width.shape,
        #     color.shape,
        #     fracture_thickness.shape,
        #     fracture_num_fractures.shape,
        #     label.shape,
        # )
        return {
            "bpd": bpd,
            "log_prob": log_prob,
            "v": [width, color, fracture_thickness, fracture_num_fractures, label],
            "v_hat": v_hat,
        }

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        log_prob = torch.cat([o["log_prob"] for o in outputs])
        # v_hat = torch.cat([o["v_hat"] for o in outputs])
        loss = -log_prob.mean()
        self.log("val_loss", loss, prog_bar=True)

        # v = torch.cat([o["v"] for o in outputs])
        # mcc = mean_correlation_coefficient(v_hat, v)
        # mcc_spearman = mean_correlation_coefficient(v_hat, v, method="spearman")

        # self.log("val_mcc", mcc.mean(), prog_bar=True)
        # for i, mcc_value in enumerate(mcc):
        #     self.log(f"val_mcc_{i}", mcc_value, prog_bar=False)
        # self.log("val_mcc_spearman", mcc_spearman.mean(), prog_bar=True)
        # for i, mcc_value in enumerate(mcc_spearman):
        #     self.log(f"val_mcc_spearman_{i}", mcc_value, prog_bar=False)

    def test_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> Union[None, dict[str, Tensor]]:
        (
            x,
            width,
            color,
            fracture_thickness,
            fracture_num_fractures,
            label,
            distr_indicators,
            intervention_targets,
        ) = batch
        log_prob = self.encoder.log_prob(
            x, y=None, env=distr_indicators, intervention_targets=intervention_targets
        )

        return {
            "log_prob": log_prob,
            "v": [width, color, fracture_thickness, fracture_num_fractures, label],
            "v_hat": self(x),
        }

    def test_epoch_end(self, outputs: List[dict]) -> None:
        log_prob = torch.cat([o["log_prob"] for o in outputs])
        loss = -log_prob.mean()

        v = torch.cat([o["v"] for o in outputs])
        v_hat = torch.cat([o["v_hat"] for o in outputs])
        mcc = mean_correlation_coefficient(v_hat, v)

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_mcc", mcc.mean(), prog_bar=False)
        for i, mcc_value in enumerate(mcc):
            self.log(f"test_mcc_{i}", mcc_value, prog_bar=False)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_hat = self.encoder(x)
        return v_hat

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        num_envs = len(self.encoder.causalq0.intervention_targets_per_distr)

        # number of variables over the cluster dag
        num_vars = self.encoder.causalq0.dag.number_of_nodes()

        # do not train any parameters that are not supposed to be trained
        # XXX: in this case, we do not update exogenous variable distributions that are fixed
        for param_idx, (distr_idx, idx) in enumerate(product(range(num_envs), range(num_vars))):
            if hasattr(self.encoder.causalq0, "noise_means_requires_grad"):
                if not self.encoder.causalq0.noise_means_requires_grad[distr_idx][idx]:
                    list(self.encoder.causalq0.noise_means.parameters())[param_idx].grad = None
                if not self.encoder.causalq0.noise_stds_requires_grad[distr_idx][idx]:
                    list(self.encoder.causalq0.noise_stds.parameters())[param_idx].grad = None
