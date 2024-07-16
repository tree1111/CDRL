import argparse
import logging
import random
from pathlib import Path

import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from gendis.datasets import CausalMNIST, ClusteredMultiDistrDataModule
from gendis.variational.vae import VAE


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        print(x.shape)
        return x.view(x.size(0), self.channels, self.height, self.width)


# Stride 2 by default
def ConvBlock(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


# Stride 2 by default
def DeconvBlock(
    in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, last=False
):
    if not last:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.Tanh(),
    )


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
    parser.add_argument("--max_epochs", type=int, default=20_000, help="Max epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cuda", help="Accelerator (cpu, cuda, mps)"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--log_dir", type=str, default="./", help="Batch size")
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
    # root = "/Users/adam2392/pytorch_data/"
    # accelerator = "cpu"
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    accelerator = args.accelerator
    batch_size = args.batch_size
    log_dir = args.log_dir

    devices = 1
    n_jobs = 1
    num_workers = 4
    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    model_fname = f"{graph_type}-seed={seed}-model.pt"

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {model_fname} \n")

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
    for intervention_idx in [
        None
         , 1, 2, 3
    ]:
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

    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 2e-4

    channels = 3
    input_height = 28
    input_width = 28
    input_shape = (channels, input_height, input_width)

    # 01: Define the encoder
    kernel_size = 1
    encoder = nn.Sequential(
        ConvBlock(channels, 28, kernel_size=kernel_size),
        ConvBlock(28, 64, 4),
        ConvBlock(64, 128, 4),
        ConvBlock(128, 256, 2),
        nn.Flatten(),
        nn.Linear(256, 3),
    )

    decoder = nn.Sequential(
        nn.Linear(3, 256 * 2 * 2),
        Stack(256, 2, 2),  # Output: (256, 2, 2)
        DeconvBlock(256, 128, 4, stride=2, padding=1, output_padding=0),  # Output: (128, 4, 4)
        DeconvBlock(128, 64, 4, stride=2, padding=1, output_padding=0),  # Output: (64, 8, 8)
        DeconvBlock(64, 28, 4, stride=2, padding=1, output_padding=0),  # Output: (28, 16, 16)
        DeconvBlock(
            28, channels, 3, stride=2, padding=3, output_padding=1, last=True
        ),  # Output: (3, 28, 28)
    )

    # 02: Define now the full pytorch lightning model
    model = VAE(
        latent_dim=3,
        encoder=encoder,
        decoder=decoder,
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
    torch.save(model, checkpoint_dir / model_fname)
