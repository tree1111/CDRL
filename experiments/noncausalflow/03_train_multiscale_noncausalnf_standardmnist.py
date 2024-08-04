# Training of a RealNVP model on the MNIST dataset with independent high-dimensional factors
# as the independent noise variables, which do not change with respect to each dataset.
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from gendis.noncausal.flows import (
    CouplingLayer,
    GatedConvNet,
    SplitFlow,
    SqueezeFlow,
    VariationalDequantization,
    create_channel_mask,
    create_checkerboard_mask,
)
from gendis.noncausal.model import ImageFlow


def generate_list(x, n_clusters):
    quotient = x // n_clusters
    remainder = x % n_clusters
    result = [quotient] * (n_clusters - 1)
    result.append(quotient + remainder)
    return result


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        # self.default_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.ToTensor(),
        #     ]
        # )

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, download=True, train=False, transform=self.transform)
        self.mnist_predict = MNIST(
            self.data_dir, download=True, train=False, transform=self.transform
        )
        mnist_full = MNIST(self.data_dir, download=True, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


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
    pass


def train_from_scratch(
    data_module,
    max_epochs,
    logger,
    devices,
    accelerator,
    checkpoint_dir,
    model_fname,
):
    flow_layers = []
    n_flows = 4
    n_chs = 1

    # add variational dequantization
    vardeq_layers = [
        CouplingLayer(
            network=GatedConvNet(c_in=n_chs * 2, c_out=n_chs * 2, c_hidden=16),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
            c_in=n_chs,
        )
        for i in range(4)
    ]
    # flow_layers += vardeq_layers
    flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]

    # flow_layers += [Dequantization()]
    # first create a sequence of channel and checkerboard masking
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs, c_hidden=24),
                mask=create_channel_mask(c_in=n_chs, invert=(i % 2 == 1)),
                c_in=n_chs,
            )
        ]
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs, c_hidden=24),
                mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                c_in=n_chs,
            )
        ]
    flow_layers += [SqueezeFlow()]
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs * 4, c_hidden=32),
                mask=create_channel_mask(c_in=n_chs * 4, invert=(i % 2 == 1)),
                c_in=n_chs * 4,
            )
        ]
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs * 4, c_hidden=32),
                mask=create_checkerboard_mask(h=14, w=14, invert=(i % 2 == 1)),
                c_in=n_chs * 4,
            )
        ]

    flow_layers += [SplitFlow(), SqueezeFlow()]
    n_chs_after_split = int(n_chs * 4 * 4 / 2)
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs_after_split, c_hidden=48),
                mask=create_channel_mask(c_in=n_chs_after_split, invert=(i % 2 == 1)),
                c_in=n_chs_after_split,
            )
        ]
    flow_layers += [SplitFlow()]
    n_chs_after_split_after_split = int(n_chs_after_split / 2)
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=n_chs_after_split_after_split, c_hidden=64),
                mask=create_channel_mask(c_in=n_chs_after_split_after_split, invert=(i % 2 == 1)),
                c_in=n_chs_after_split_after_split,
            )
        ]
    # flow_layers += [Reshape((24, 7, 7), (784 * 3 // 2,))]
    print("\n\nRunning forward direction...")
    output, ldj = torch.randn(5, n_chs, 28, 28), 0
    output = output - output.min()
    output = output / output.max() * 255
    for flow in flow_layers:
        output, ldj = flow(output, ldj)
        print("Running: ", type(flow), output.shape)

    # Sample latent representation from prior
    # if z_init is None:
    #     z = self.prior.sample(sample_shape=img_shape).to(self.device)
    # else:
    #     z = z_init.to(self.device)

    # # Transform z to x by inverting the flows
    # ldj = torch.zeros(img_shape[0], device=self.device)
    # for flow in reversed(self.flows):
    #     z, ldj = flow(z, ldj, reverse=True)

    model = ImageFlow(
        flow_layers,
        # prior=noiseq0,
        lr=lr,
        lr_min=lr_min,
        lr_scheduler=lr_scheduler,
    )
    # print(output.shape)
    # print('\n\n Now running reverse')
    # for flow in reversed(flow_layers):
    #     output, ldj = flow(output, ldj, reverse=True)
    #     print("Running: ", type(flow), output.shape)
    # prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    output = model.sample((5, n_chs_after_split_after_split, 7, 7))
    print(output.shape)
    # assert False

    # 04b: Define the trainer for the model
    checkpoint_dir = Path(checkpoint_root_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    logger = None
    wandb = False
    check_val_every_n_epoch = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor="val_bpd",
        every_n_epochs=check_val_every_n_epoch,
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=devices,
        gradient_clip_val=gradient_clip_val,
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
    num_workers = 10
    gradient_clip_val = None  # 1.0
    batch_size = args.batch_size
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 1e-3

    # root = "/Users/adam2392/pytorch_data/"
    # accelerator = "cpu"
    # num_workers = 1
    # batch_size = 10
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    log_dir = args.log_dir

    devices = 1
    n_jobs = 1

    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    model_id_fname = f"vardeq-discretize-normalMNIST-batch{batch_size}-{graph_type}-seed={seed}"
    checkpoint_root_dir = Path(f"nf-{model_id_fname}")
    model_fname = f"nf-{model_id_fname}-model.pt"

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {model_fname} \n")

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    # since MNIST images are in [0, 1], we need to discretize them
    def discretize(sample):
        return (sample * 255).to(torch.int32)

    # set up transforms for each image to augment the dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize((32, 32)),
            # nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            # nf.utils.Jitter(1 / 256.0),  # apply random generation
            discretize,
            torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    # load in MNIST
    # mnist_train = MNIST(root, train=True, download=True, transform=transform)
    # mnist_train = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers)
    data_module = MNISTDataModule(data_dir=root, batch_size=batch_size, transform=transform)

    train_from_scratch(
        data_module,
        max_epochs,
        logger,
        devices,
        accelerator,
        checkpoint_root_dir,
        model_fname,
    )
