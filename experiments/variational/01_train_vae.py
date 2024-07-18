# refs:
# 1. https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py for VAE with resnet

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
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

from gendis.datasets import CausalMNIST, ClusteredMultiDistrDataModule
from gendis.variational.vae import VAE


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
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
        nn.Sigmoid(),
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
    parser.add_argument("--max_epochs", type=int, default=10_000, help="Max epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cuda", help="Accelerator (cpu, cuda, mps)"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--log_dir", type=str, default="./", help="Batch size")
    return parser


def train_from_checkpoint(
    data_module,
    max_epochs,
    logger,
    devices,
    accelerator,
    checkpoint_path,
    checkpoint_dir,
    model_fname,
):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    current_max_epochs = checkpoint["epoch"]
    max_epochs += current_max_epochs
    model = VAE.load_from_checkpoint(checkpoint_path)

    # 04b: Define the trainer for the model
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


def train_from_scratch(
    data_module,
    max_epochs,
    logger,
    devices,
    accelerator,
    checkpoint_dir,
    model_fname,
):
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 2e-5

    channels = 3
    input_height = 32
    input_width = 32
    input_shape = (channels, input_height, input_width)

    # 01: Define the Convolutional encoder/decoder
    # kernel_size = 1
    # encoder = nn.Sequential(
    #     ConvBlock(channels, 28, kernel_size=kernel_size),
    #     ConvBlock(28, 64, 4),
    #     ConvBlock(64, 128, 4),
    #     ConvBlock(128, 256, 2),
    #     nn.Flatten(),
    #     nn.Linear(256, 3),
    # )
    # decoder = nn.Sequential(
    #     nn.Linear(3, 256 * 2 * 2),
    #     Stack(256, 2, 2),  # Output: (256, 2, 2)
    #     DeconvBlock(256, 128, 4, stride=2, padding=1, output_padding=0),  # Output: (128, 4, 4)
    #     DeconvBlock(128, 64, 4, stride=2, padding=1, output_padding=0),  # Output: (64, 8, 8)
    #     DeconvBlock(64, 28, 4, stride=2, padding=1, output_padding=0),  # Output: (28, 16, 16)
    #     DeconvBlock(
    #         28, channels, 3, stride=2, padding=3, output_padding=1, last=True
    #     ),  # Output: (3, 28, 28)
    # )

    encoder = resnet18_encoder(False, False)
    decoder = resnet18_decoder(
        latent_dim=latent_dim, input_height=input_height, first_conv=False, maxpool1=False
    )
    # encoder = nn.Sequential(
    #     nn.Linear(2352, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 256),
    # )

    # 02: Define now the full pytorch lightning model
    model = VAE(
        latent_dim=3,
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=512,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
    )

    # 04b: Define the trainer for the model
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


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()

    graph_type = "chain"
    adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    latent_dim = len(adjacency_matrix)

    root = "/home/adam2392/projects/data/"
    accelerator = args.accelerator
    intervention_types = [None, 1, 2, 3]
    # root = "/Users/adam2392/pytorch_data/"
    # accelerator = "cpu"
    # intervention_types = [None]
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    log_dir = args.log_dir

    devices = 1
    n_jobs = 1
    num_workers = 4
    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    model_fname = f"vae-resnet-{graph_type}-seed={seed}-model.pt"
    checkpoint_dir = Path(f"./vae-resnet-{graph_type}-seed={seed}")

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
            torchvision.transforms.Resize((32, 32)),
            nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            nf.utils.Jitter(1 / 256.0),  # apply random generation
            torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    # now we can wrap this in a pytorch lightning datamodule
    data_module = ClusteredMultiDistrDataModule(
        root=root,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        intervention_types=intervention_types,
        transform=transform,
        log_dir=log_dir,
        flatten=False,
    )
    # data_module.setup()

    # train_from_scratch(
    #     data_module,
    #     max_epochs,
    #     logger,
    #     devices,
    #     accelerator,
    #     checkpoint_dir,
    #     model_fname,
    # )

    epoch = 9988
    step = 419538
    checkpoint_path = checkpoint_dir / f"epoch={epoch}-step={step}.ckpt"
    train_from_checkpoint(
        data_module,
        max_epochs,
        logger,
        devices,
        accelerator,
        checkpoint_path,
        checkpoint_dir,
        model_fname,
    )
