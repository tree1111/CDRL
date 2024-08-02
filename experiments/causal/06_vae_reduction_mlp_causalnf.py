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
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint

from gendis.causal.modelv3 import CausalFlowModel, CausalNormalizingFlow
from gendis.datasets import MultiDistrDataModule
from gendis.normalizing_flow.distribution import ClusteredCausalDistribution


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
    # training misc args
    parser.add_argument("--root_dir", type=str, default="./", help="Root directory")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--max_epochs", type=int, default=20_000, help="Max epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cuda", help="Accelerator (cpu, cuda, mps)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
    monitor = "train_kld"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    current_max_epochs = checkpoint["epoch"]
    max_epochs += current_max_epochs
    # model = ImageFlow.load_from_checkpoint(checkpoint_path)

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


def make_model(adjacency_matrix, intervention_targets_per_distr):
    # Define flows
    # Define list of flows
    K = 32
    net_hidden_layers = 3
    net_hidden_dim = 128
    latent_dim = 32

    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_dim, net_hidden_layers, net_hidden_dim
            )
        ]

    latent_shape = (32,)
    q0 = ClusteredCausalDistribution(
        adjacency_matrix=adjacency_matrix,
        cluster_sizes=generate_list(np.prod(latent_shape), latent_dim),
        input_shape=latent_shape,
        intervention_targets_per_distr=torch.vstack(intervention_targets_per_distr),
        hard_interventions_per_distr=None,
    )

    # Construct flow model with the multiscale architecture
    model = CausalNormalizingFlow(q0, flows)
    return model


def train_from_scratch(
    data_module,
    max_epochs,
    devices,
    accelerator,
    checkpoint_dir,
    model_fname,
    adjacency_matrix,
    intervention_targets_per_distr,
):
    lr_scheduler = "cosine"
    lr_min = 1e-7
    lr = 1e-3

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\n\n\tsaving to {model_fname} \n")

    # Define the distributions
    # Sample latent representation from prior
    # if z_init is None:
    #     z = self.prior.sample(sample_shape=img_shape).to(self.device)
    # else:
    #     z = z_init.to(self.device)

    # # Transform z to x by inverting the flows
    # ldj = torch.zeros(img_shape[0], device=self.device)
    # for flow in reversed(self.flows):
    #     z, ldj = flow(z, ldj, reverse=True)

    nf_flow = make_model(
        adjacency_matrix=adjacency_matrix,
        intervention_targets_per_distr=intervention_targets_per_distr,
    )
    model = CausalFlowModel(model=nf_flow, lr=lr, lr_min=lr_min, lr_scheduler=lr_scheduler)

    print("\n\nRunning forward direction...")
    output, ldj = torch.randn(batch_size, 32), 0
    output = nf_flow.forward(output)
    # output
    # output = output - output.min()
    # output = output / output.max() * 255
    # for idx, flow in enumerate(nf_flow.flows):
    #     # try:
    #     #     print(flow.batch_dims, flow.n_dim, flow.s.shape)
    #     # except Exception as e:
    #     #     print(idx)
    #     output, ldj = flow(output, ldj)
    #     print("Running: ", type(flow), output.shape, ldj.shape)

    print(output.shape)
    print("\n\n Now running reverse")
    output = nf_flow.inverse(output)
    print(output.shape)
    # for flow in reversed(nf_flow.flows):
    #     output, ldj = flow(output, ldj)
    #     print("Running: ", type(flow), output.shape)
    # prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    output, ldj = model.sample(num_samples=16)
    print(output.shape, ldj.shape)

    # 04b: Define the trainer for the model
    checkpoint_dir = Path(checkpoint_root_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    wandb = False
    logger = None
    check_val_every_n_epoch = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
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
    # accelerator = "mps"
    # intervention_types = [None, 1]
    # num_workers = 1
    # batch_size = 10

    root = Path(root)
    new_root = root / "causalbar_reduction_dat/"
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    log_dir = args.log_dir

    devices = 1
    n_jobs = 1

    print("Running with n_jobs:", n_jobs)

    # output filename for the results
    model_name = f"mlp-nf-onvae-reduction-cosinelr-batch{batch_size}-{graph_type}-seed={seed}"
    checkpoint_root_dir = Path(model_name)
    model_fname = f"{model_name}-model.pt"

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    # set up transforms for each image to augment the dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize((32, 32)),
            # nf.utils.Scale(255.0 / 256.0),  # normalize the pixel values
            # nf.utils.Jitter(1 / 256.0),  # apply random generation
            # torchvision.transforms.RandomRotation(350),  # get random rotations
        ]
    )

    # demo to load dataloader. please make sure transform is None. d
    data_module = MultiDistrDataModule(
        root=new_root,
        graph_type="chain",
        batch_size=batch_size,
        stratify_distrs=True,
        transform=None,
        num_workers=num_workers,
    )
    data_module.setup()

    intervention_targets_per_distr = []
    # print(torch.vstack(data_module.dataset.labels).shape)
    # print(torch.vstack(data_module.dataset.intervention_targets).shape)
    print(data_module.dataset.intervention_targets.shape)
    print(data_module.dataset.labels.shape)
    for distr_idx in data_module.dataset.distribution_idx.unique():
        idx = np.argwhere(data_module.dataset.distribution_idx == distr_idx)[0][0]
        intervention_targets_per_distr.append(data_module.dataset.intervention_targets[idx])
    print(idx)

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
        devices,
        accelerator,
        checkpoint_root_dir,
        model_fname,
        adjacency_matrix,
        intervention_targets_per_distr,
    )
