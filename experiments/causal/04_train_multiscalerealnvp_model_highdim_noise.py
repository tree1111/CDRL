# Training of a RealNVP model on the MNIST dataset with independent high-dimensional factors
# as the independent noise variables, which do not change with respect to each dataset.
import argparse
import logging
import random
from pathlib import Path

import normflows as nf
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from gendis.datasets import MultiDistrDataModule
from gendis.noncausal.flows import (
    ActNorm,
    CouplingLayer,
    GatedConvNet,
    SplitFlow,
    SqueezeFlow,
    create_channel_mask,
    create_checkerboard_mask,
)
from gendis.noncausal.model import ImageFlow


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


def generate_list(x, n_clusters):
    quotient = x // n_clusters
    remainder = x % n_clusters
    result = [quotient] * (n_clusters - 1)
    result.append(quotient + remainder)
    return result


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
    monitor = "val_bpd"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    current_max_epochs = checkpoint["epoch"]
    max_epochs += current_max_epochs
    model = ImageFlow.load_from_checkpoint(checkpoint_path)

    # 04b: Define the trainer for the model
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    logger = None
    wandb = False
    check_val_every_n_epoch = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor=monitor,
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
    intervention_targets_per_distr = data_module.dataset.intervention_targets
    hard_interventions_per_distr = None

    graph = adjacency_matrix
    cluster_sizes = generate_list(784 * 3, 3)
    cluster_sizes = None
    input_shape = (3, 28, 28)
    channels = 3
    use_vardeq = True

    # Define the distributions
    noiseq0 = nf.distributions.DiagGaussian(shape=(784 * 3,))

    flow_layers = []
    n_flows = 4
    # first create a sequence of channel and checkerboard masking
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=3, c_hidden=24),
                mask=create_channel_mask(c_in=3, invert=(i % 2 == 1)),
                c_in=3,
            )
        ]
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=3, c_hidden=24),
                mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                c_in=3,
            )
        ]
        # input the shape of the coupling layer
        flow_layers += [ActNorm((3, 1, 1))]
    flow_layers += [SqueezeFlow()]
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=12, c_hidden=32),
                mask=create_channel_mask(c_in=12, invert=(i % 2 == 1)),
                c_in=12,
            )
        ]
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=12, c_hidden=32),
                mask=create_checkerboard_mask(h=14, w=14, invert=(i % 2 == 1)),
                c_in=12,
            )
        ]
        # input the shape of the coupling layer
        flow_layers += [ActNorm((3, 1, 1))]

    flow_layers += [SplitFlow(), SqueezeFlow()]
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=24, c_hidden=48),
                mask=create_channel_mask(c_in=24, invert=(i % 2 == 1)),
                c_in=24,
            )
        ]
        # input the shape of the coupling layer
        flow_layers += [ActNorm((3, 1, 1))]
    flow_layers += [SplitFlow()]
    for i in range(n_flows):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=12, c_hidden=64),
                mask=create_channel_mask(c_in=12, invert=(i % 2 == 1)),
                c_in=12,
            )
        ]
        # input the shape of the coupling layer
        flow_layers += [ActNorm((3, 1, 1))]
    # flow_layers += [Reshape((24, 7, 7), (784 * 3 // 2,))]
    print("\n\nRunning forward direction...")
    output, ldj = torch.randn(batch_size, 3, 28, 28), 0
    for idx, flow in enumerate(flow_layers):
        # try:
        #     print(flow.batch_dims, flow.n_dim, flow.s.shape)
        # except Exception as e:
        #     print(idx)
        output, ldj = flow(output, ldj)
        print("Running: ", type(flow), output.shape, ldj.shape)

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
    output = model.sample((5, 12, 7, 7))
    print(output.shape)

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
    intervention_types = [None, 1, 2, 3]
    num_workers = 10
    gradient_clip_val = None  # 1.0
    batch_size = args.batch_size
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 1e-3

    # root = "/Users/adam2392/pytorch_data/"
    # accelerator = "cpu"
    # intervention_types = [None, 1]
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
    checkpoint_root_dir = Path(
        f"nf-actnorm-3point1M-cosinelr-batch{batch_size}-{graph_type}-seed={seed}"
    )
    model_fname = (
        f"nf-actnorm-3point1M-cosinelr-batch{batch_size}-{graph_type}-seed={seed}-model.pt"
    )

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
            # torchvision.transforms.Resize((32, 32)),
            nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            nf.utils.Jitter(1 / 256.0),  # apply random generation
            torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    # now we can wrap this in a pytorch lightning datamodule
    # data_module = ClusteredMultiDistrDataModule(
    #     root=root,
    #     graph_type=graph_type,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     intervention_types=intervention_types,
    #     transform=transform,
    #     log_dir=log_dir,
    #     flatten=False,
    # )
    data_module = MultiDistrDataModule(
        root=root,
        stratify_distrs=True,
        graph_type=graph_type,
        num_workers=num_workers,
        batch_size=batch_size,
        transform=transform,
        log_dir=log_dir,
    )
    data_module.setup()

    # epoch = 9306
    # step = 781788
    # checkpoint_path = checkpoint_root_dir / f"epoch={epoch}-step={step}.ckpt"
    # train_from_checkpoint(
    #     data_module,
    #     max_epochs,
    #     logger,
    #     devices,
    #     accelerator,
    #     checkpoint_path,
    #     checkpoint_root_dir,
    #     model_fname,
    # )

    # train from scratch
    train_from_scratch(
        data_module,
        max_epochs,
        logger,
        devices,
        accelerator,
        checkpoint_root_dir,
        model_fname,
    )
