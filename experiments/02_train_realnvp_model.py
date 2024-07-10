import argparse
import logging
import random
from pathlib import Path

import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from gendis.datasets import CausalMNIST, ClusteredMultiDistrDataModule
from gendis.encoder import CausalMultiscaleFlow, NonparametricClusteredCausalEncoder
from gendis.model import NeuralClusteredASCMFlow
from gendis.normalizing_flow.distribution import NonparametricClusteredCausalDistribution


def generate_list(x, n_clusters):
    quotient = x // n_clusters
    remainder = x % n_clusters
    result = [quotient] * (n_clusters - 1)
    result.append(quotient + remainder)
    return result


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    # parser.add_argument(
    #     "--dataset_name", type=str, default="rotten_tomatoes", help="name of dataset"
    # )
    # parser.add_argument(
    #     "--subsample_frac", type=float, default=1, help="fraction of samples to use"
    # )

    # training misc args
    parser.add_argument("--root_dir", type=str, default="./", help="Root directory")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cuda", help="Accelerator (cpu, cuda, mps)"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--log_dir", type=str, default="./", help="Batch size")

    # model args
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     choices=["decision_tree", "ridge"],
    #     default="decision_tree",
    #     help="name of model",
    # )
    # parser.add_argument("--alpha", type=float, default=1, help="regularization strength")
    # parser.add_argument("--max_depth", type=int, default=2, help="max depth of tree")
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()

    graph_type = "chain"
    adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    latent_dim = len(adjacency_matrix)
    results_dir = Path("./results/")
    results_dir.mkdir(exist_ok=True, parents=True)

    root = "/home/adam2392/projects/data/"
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    accelerator = args.accelerator
    batch_size = args.batch_size
    log_dir = args.log_dir

    devices = 2
    n_jobs = 1
    num_workers = 2
    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    fname = results_dir / f"{graph_type}-seed={seed}-results.npz"

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {fname} \n")

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    # set up transforms for each image to augment the dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            nf.utils.Jitter(1 / 256.0),  # apply random generation
            torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    # load dataset
    datasets = []
    intervention_targets_per_distr = []
    hard_interventions_per_distr = None
    num_distrs = 0
    for intervention_idx in [None, 1, 2, 3]:
        dataset = CausalMNIST(
            root=root,
            graph_type=graph_type,
            label=0,
            download=True,
            train=True,
            n_jobs=None,
            intervention_idx=intervention_idx,
            transform=transform,
        )
        dataset.prepare_dataset(overwrite=False)
        datasets.append(dataset)
        num_distrs += 1
        intervention_targets_per_distr.append(dataset.intervention_targets)

    # now we can wrap this in a pytorch lightning datamodule
    data_module = ClusteredMultiDistrDataModule(
        datasets=datasets,
        num_workers=num_workers,
        batch_size=batch_size,
        intervention_targets_per_distr=intervention_targets_per_distr,
        log_dir=log_dir,
        flatten=False,
    )
    data_module.setup()

    n_flows = 3  # number of flows to use in nonlinear ICA model
    lr_scheduler = None
    lr_min = 0.0
    lr = 1e-6

    # Define the model
    net_hidden_dim = 128
    net_hidden_dim_cbn = 128
    net_hidden_layers = 3
    net_hidden_layers_cbn = 3
    fix_mechanisms = False

    graph = adjacency_matrix
    cluster_sizes = generate_list(784 * 3, 3)

    # 01: Define the causal base distribution with the graph
    causalq0 = NonparametricClusteredCausalDistribution(
        adjacency_matrix=graph,
        cluster_sizes=cluster_sizes,
        intervention_targets_per_distr=intervention_targets_per_distr,
        hard_interventions_per_distr=hard_interventions_per_distr,
        fix_mechanisms=fix_mechanisms,
        n_flows=n_flows,
        n_hidden_dim=net_hidden_dim,
        n_layers=net_hidden_layers,
    )

    input_shape = (3, 28, 28)
    channels = 3

    # Define flows
    L = 2
    K = 3
    n_dims = np.prod(input_shape)
    hidden_channels = 256
    split_mode = "channel"
    scale = True

    stride_factor = 2

    # Set up flows, distributions and merge operations
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            n_chs = channels * 2 ** (L + 1 - i)
            flows_ += [
                nf.flows.GlowBlock(n_chs, hidden_channels, split_mode=split_mode, scale=scale)
            ]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (
                input_shape[0] * stride_factor ** (L - i),
                input_shape[1] // stride_factor ** (L - i),
                input_shape[2] // stride_factor ** (L - i),
            )
        else:
            latent_shape = (
                input_shape[0] * stride_factor ** (L + 1),
                input_shape[1] // stride_factor**L,
                input_shape[2] // stride_factor**L,
            )
        print(n_chs, np.prod(latent_shape), latent_shape)

    # 03: Define the final normalizing flow model
    # Construct flow model with the multiscale architecture
    encoder = CausalMultiscaleFlow(causalq0, flows, merges)

    # 04a: Define now the full pytorch lightning model
    model = NeuralClusteredASCMFlow(
        encoder=encoder,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
    )

    # 04b: Define the trainer for the model
    checkpoint_root_dir = f"{graph_type}-seed={seed}"
    checkpoint_dir = Path(checkpoint_root_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    logger = None
    wandb = False
    check_val_every_n_epoch = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor="train_loss",
        every_n_epochs=check_val_every_n_epoch,
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=devices,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
    )

    # 05: Fit the model and save the data
    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    torch.save(model, checkpoint_dir / "model.pt")
