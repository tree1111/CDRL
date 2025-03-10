# Using the normflows package, train a normaliznig flow model on a dataset of colored images.# Training of a RealNVP model on the MNIST dataset with independent high-dimensional factors
# as the independent noise variables, which do not change with respect to each dataset.
import argparse
import logging
import random
from pathlib import Path

import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from gendis.causal.modelv3 import CausalFlowModel, CausalNormalizingFlow
from gendis.datasets import MultiDistrDataModule
from gendis.normalizing_flow.distribution import ClusteredCausalDistribution


def make_model(adjacency_matrix, intervention_targets_per_distr, confounded_vars=None):
    # Define flows
    # Define list of flows
    L = 2
    K = 32
    net_hidden_layers = 3
    net_hidden_dim = 64
    latent_dim = 32

    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_dim, net_hidden_layers, net_hidden_dim
            )
        ]

    latent_shape = (32,)
    # q0 = ClusteredCausalDistribution(
    #     adjacency_matrix=adjacency_matrix,
    #     cluster_sizes=generate_list(np.prod(latent_shape), latent_dim),
    #     input_shape=latent_shape,
    #     intervention_targets_per_distr=torch.vstack(intervention_targets_per_distr),
    #     hard_interventions_per_distr=None,
    # )

    # independent noise with causal prior
    q0 = ClusteredCausalDistribution(
        adjacency_matrix=adjacency_matrix,
        cluster_sizes=[8, 8, 8],
        input_shape=latent_shape,
        ind_noise_dim=8,
        intervention_targets_per_distr=torch.vstack(intervention_targets_per_distr),
        hard_interventions_per_distr=None,
        confounded_variables=confounded_vars,
    )

    # Construct flow model with the multiscale architecture
    model = CausalNormalizingFlow(q0, flows)
    return model


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    # training misc args
    parser.add_argument("--root_dir", type=str, default="./", help="Root directory")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--max_epochs", type=int, default=20_000, help="Max epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cuda", help="Accelerator (cpu, cuda, mps)"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--log_dir", type=str, default="./", help="Batch size")
    return parser


def train_from_scratch(
    data_module,
    max_epochs,
    devices,
    accelerator,
    checkpoint_dir,
    model_fname,
    confounded_vars=None,
):
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 1e-3
    n_chs = 3
    height = 28
    width = 28
    hidden_size = 32

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {model_fname} \n")

    nf_flow = make_model(
        adjacency_matrix=adjacency_matrix,
        intervention_targets_per_distr=intervention_targets_per_distr,
        confounded_vars=confounded_vars,
    )

    model = CausalFlowModel(model=nf_flow, lr=lr, lr_min=lr_min, lr_scheduler=lr_scheduler)

    # 04b: Define the trainer for the model
    checkpoint_dir = Path(checkpoint_root_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    logger = None
    check_val_every_n_epoch = 1
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor="train_kld",
        every_n_epochs=check_val_every_n_epoch,
    )

    # Train the model
    trainer = pl.Trainer(
        logger=logger,
        devices=devices,
        gradient_clip_val=gradient_clip_val,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        max_epochs=max_epochs,
        # max_epochs=2,
        # max_steps=3,
        # fast_dev_run=3,
    )

    # 05: Fit the model and save the data
    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    torch.save(model, checkpoint_dir / model_fname)


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()

    graph_type = "nonmarkov"
    adjacency_matrix = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    confounded_vars = [[0, 2]]  # confounding between digit and color-bar
    latent_dim = len(adjacency_matrix)

    root = "/home/adam2392/projects/data/"
    accelerator = args.accelerator
    intervention_types = [None, 1, 2, 3, 4]
    num_workers = 10
    gradient_clip_val = None  # 1.0
    batch_size = args.batch_size
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 1e-3
    dataset_clip = None

    # root = "/Users/adam2392/pytorch_data/"
    # accelerator = "mps"
    # intervention_types = [None, 1]
    # num_workers = 1
    # batch_size = 10
    # dataset_clip = 1000

    root = Path(root)
    # XXX: change this depending on the dataset
    new_root = root / "vae-reduction"
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    log_dir = args.log_dir

    devices = 1
    n_jobs = 1

    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    model_name = f"nfonvae-reduction-cosinelr-batch{batch_size}-{graph_type}-seed={seed}"
    checkpoint_root_dir = Path(model_name)
    model_fname = f"{model_name}-model.pt"

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    # set up transforms for each image to augment the dataset
    transform = None

    # demo to load dataloader. please make sure transform is None. d
    data_module = MultiDistrDataModule(
        root=new_root,
        graph_type=graph_type,
        batch_size=batch_size,
        stratify_distrs=True,
        transform=transform,
        num_workers=num_workers,
        dataset_name="digitcolorbar",
        subsample=dataset_clip,
    )
    data_module.setup()

    intervention_targets_per_distr = []
    print(data_module.dataset.intervention_targets.shape)
    print(data_module.dataset.labels.shape)
    for distr_idx in data_module.dataset.distribution_idx.unique():
        idx = np.argwhere(data_module.dataset.distribution_idx == distr_idx)[0][0]
        intervention_targets_per_distr.append(data_module.dataset.intervention_targets[idx])
    print(idx)

    unique_rows = np.unique(data_module.dataset.intervention_targets, axis=0)
    print("Unique intervention targets: ", unique_rows)

    # train from scratch
    train_from_scratch(
        data_module,
        max_epochs,
        devices,
        accelerator,
        checkpoint_root_dir,
        model_fname,
        confounded_vars=confounded_vars,
    )
