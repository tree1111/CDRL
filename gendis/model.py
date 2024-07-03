from __future__ import annotations

from itertools import product
from typing import Optional, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from .metrics import mean_correlation_coefficient
from .encoder import (
    ClusteredCausalEncoder,
    NaiveClusteredCausalEncoder,
    NonparametricClusteredCausalEncoder,
    ParametricClusteredCausalEncoder,
)


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


class BaseNeuralClusteredASCMFlow(pl.LightningModule):
    """A Neural Clustered Augmented SCM Flow Model.

    Base class for Neural augmented structural causal models (NASCM-Flow). It implements the
    training loop and the evaluation metrics.

    The model is an encoder-decoder model where the encoding and decoding uses flows as the
    layers (i.e. invertible transformations).

    Attributes
    ----------
    graph : np.ndarray, shape (num_nodes, num_nodes)
        Adjacency matrix of the causal graph assumed by the model. This is not necessarily
        the true adjacency matrix of the data generating process (see below).
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.
    encoder : ClusteredCausalEncoder
        The causal encoder. Needs to be set in subclasses. The inverse of the encoder is the
        unmixing function.

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

    encoder: ClusteredCausalEncoder  # set in subclasses

    def __init__(
        self,
        graph: np.ndarray,
        cluster_sizes: List[int] = None,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.cluster_sizes = cluster_sizes
        self.graph = graph

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min

        self.save_hyperparameters()

    @property
    def latent_dim(self):
        return np.sum(self.cluster_sizes) if self.cluster_sizes is not None else self.graph.shape[0]

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
        log_prob, res = self.encoder.multi_env_log_prob(x, distr_indicators, intervention_targets)
        loss = -log_prob.mean()

        self.log(f"train_loss (batch={batch_idx})", loss, prog_bar=False)
        return loss

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
        log_prob, res = self.encoder.multi_env_log_prob(x, distr_indicators, intervention_targets)

        v_hat = self(x)
        print(
            log_prob.shape,
            x.shape,
            width.shape,
            color.shape,
            fracture_thickness.shape,
            fracture_num_fractures.shape,
            label.shape,
        )
        return {
            "log_prob": log_prob,
            "v": [width, color, fracture_thickness, fracture_num_fractures, label],
            "v_hat": v_hat,
        }

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        log_prob = torch.cat([o["log_prob"] for o in outputs])
        v_hat = torch.cat([o["v_hat"] for o in outputs])
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
        log_prob, res = self.encoder.multi_env_log_prob(x, distr_indicators, intervention_targets)

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
        num_envs = len(self.encoder.intervention_targets_per_distr)
        num_vars = self.graph.shape[0]

        # do not train any parameters that are not supposed to be trained
        # XXX: in this case, we do not update exogenous variable distributions that are fixed
        for param_idx, (distr_idx, idx) in enumerate(product(range(num_envs), range(num_vars))):
            if hasattr(self.encoder.q0, "noise_means_requires_grad"):
                if not self.encoder.q0.noise_means_requires_grad[distr_idx][idx]:
                    list(self.encoder.q0.noise_means.parameters())[param_idx].grad = None
                if not self.encoder.q0.noise_stds_requires_grad[distr_idx][idx]:
                    list(self.encoder.q0.noise_stds.parameters())[param_idx].grad = None


class NonlinearNeuralClusteredASCMFlow(BaseNeuralClusteredASCMFlow):
    """Nonlinear Neural Clustered Augmented SCM Flow Model.

    Class for nonlinear Neural augmented structural causal models (NASCM-Flow). It implements the
    training loop and the evaluation metrics.

    The model is an encoder-decoder model where the encoding and decoding uses flows as the
    layers (i.e. invertible transformations).

    Attributes
    ----------
    graph : np.ndarray, shape (num_nodes, num_nodes)
        Adjacency matrix of the causal graph assumed by the model. This is not necessarily
        the true adjacency matrix of the data generating process (see below).
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
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.
    n_flows : int
        Number of flows to use in the nonlinear unmixing function. Default: 1.
    n_hidden_dim : int
        Hidden dimension of the neural network used in the nonlinear unmixing function. Default: 128.
    n_layers : int
        Number of hidden layers of the neural network used in the nonlinear unmixing function. Default: 3.
    encoder : NonparametricClusteredCausalEncoder
        The causal encoder. The inverse of the encoder is the
        unmixing function.
    """

    def __init__(
        self,
        graph: np.ndarray,
        cluster_sizes: List[int] = None,
        intervention_targets_per_distr: Optional[torch.Tensor] = None,
        hard_interventions_per_distr: Optional[Tensor] = None,
        fix_mechanisms: bool = False,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        n_flows: int = 1,
        n_hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__(
            graph=graph,
            cluster_sizes=cluster_sizes,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
        )
        self.encoder = NonparametricClusteredCausalEncoder(
            self.graph,
            cluster_sizes=self.cluster_sizes,
            intervention_targets_per_distr=intervention_targets_per_distr,
            hard_interventions_per_distr=hard_interventions_per_distr,
            fix_mechanisms=fix_mechanisms,
            n_flows=n_flows,
            n_hidden_dim=n_hidden_dim,
            n_layers=n_layers,
        )
        self.save_hyperparameters()


class LinearNeuralClusteredASCMFlow(BaseNeuralClusteredASCMFlow):
    """
    CauCA model with linear unmixing function.
    """

    def __init__(
        self,
        graph: np.ndarray,
        cluster_sizes: List[int] = None,
        intervention_targets_per_distr: Tensor = None,
        hard_interventions_per_distr: Tensor = None,
        fix_mechanisms: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
    ) -> None:
        super().__init__(
            graph=graph,
            cluster_sizes=cluster_sizes,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
        )
        self.encoder = ParametricClusteredCausalEncoder(
            self.graph,
            cluster_sizes=cluster_sizes,
            intervention_targets_per_distr=intervention_targets_per_distr,
            hard_interventions_per_distr=hard_interventions_per_distr,
            fix_mechanisms=fix_mechanisms,
        )
        self.save_hyperparameters()


class NaiveNeuralClusteredASCMFlow(BaseNeuralClusteredASCMFlow):
    """
    Naive CauCA model with nonlinear unmixing function. It assumes no causal dependencies.
    """

    def __init__(
        self,
        graph: np.ndarray,
        cluster_sizes: List[int] = None,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        intervention_targets_per_distr: Optional[torch.Tensor] = None,
        hard_interventions_per_distr: Optional[Tensor] = None,
        fix_mechanisms: bool = False,
        n_flows: int = 1,
        n_hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__(
            graph=graph,
            cluster_sizes=cluster_sizes,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
        )

        self.encoder = NaiveClusteredCausalEncoder(
            self.graph,
            self.cluster_sizes,
            intervention_targets_per_distr=intervention_targets_per_distr,
            hard_interventions_per_distr=hard_interventions_per_distr,
            fix_mechanisms=fix_mechanisms,
            n_flows=n_flows,
            n_hidden_dim=n_hidden_dim,
            n_layers=n_layers,
        )
        self.save_hyperparameters()
