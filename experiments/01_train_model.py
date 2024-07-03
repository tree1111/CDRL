from pathlib import Path
import argparse
import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl

from gendis.datasets import CausalMNIST, ClusteredMultiDistrDataModule
from gendis.model import NonlinearNeuralClusteredASCMFlow


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

    root = "/Users/adam2392/pytorch_data/"
    print(args)
    # root = args.root_dir
    seed = args.seed
    max_epochs = args.max_epochs
    accelerator = args.accelerator
    batch_size = args.batch_size
    log_dir = args.log_dir

    devices = 1
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

    # load dataset
    datasets = []
    intervention_targets_per_distr = []
    hard_interventions_per_distr = None
    for intervention_idx in [None, 1, 2, 3]:
        dataset = CausalMNIST(
            root=root,
            graph_type=graph_type,
            label=0,
            download=True,
            train=True,
            n_jobs=None,
            intervention_idx=intervention_idx,
        )
        dataset.prepare_dataset(overwrite=False)
        datasets.append(dataset)

        intervention_targets_per_distr.append(dataset.intervention_targets)

    # now we can wrap this in a pytorch lightning datamodule
    data_module = ClusteredMultiDistrDataModule(
        datasets=datasets,
        num_workers=num_workers,
        batch_size=batch_size,
        intervention_targets_per_distr=intervention_targets_per_distr,
        log_dir=log_dir,
        flatten=True,
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
    model = NonlinearNeuralClusteredASCMFlow(
        cluster_sizes=generate_list(784*3, 3),
        graph=adjacency_matrix,
        intervention_targets_per_distr=intervention_targets_per_distr,
        hard_interventions_per_distr=hard_interventions_per_distr,
        n_flows=n_flows,
        n_hidden_dim=net_hidden_dim,
        n_layers=net_hidden_layers,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        fix_mechanisms=fix_mechanisms,
    )
    checkpoint_root_dir = f"{graph_type}-seed={seed}"
    checkpoint_dir = Path(checkpoint_root_dir) / f"seed-{seed}_{graph_type}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    logger = None
    wandb = False
    check_val_every_n_epoch = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=3,
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
    trainer.fit(
        model,
        datamodule=data_module,
    )

    # save the final model
    torch.save(model, checkpoint_dir / "model.pt")
    # print(f"Checkpoint dir: {checkpoint_dir}")
    # trainer.test(datamodule=data_module)

    # # save the output
    # # x, v, u, e, int_target, log_prob_gt = data_module.test_dataset[:]
    # print(x.shape)
    # print(e.shape)

    # # Step 1: Obtain learned representations, which are "predictions
    # vhat = model.forward(x)
    # corr_arr_v_vhat = np.zeros((latent_dim, latent_dim))
    # for idx in range(latent_dim):
    #     for jdx in range(latent_dim):
    #         corr_arr_v_vhat[jdx, idx] = mean_correlation_coefficient(vhat[:, (idx,)], v[:, (jdx,)])
    # print("Saving file to: ", fname)
    # np.savez_compressed(
    #     fname,
    #     x=x.detach().numpy(),
    #     v=v.detach().numpy(),
    #     u=u.detach().numpy(),
    #     e=e.detach().numpy(),
    #     int_target=int_target.detach().numpy(),
    #     log_prob_gt=log_prob_gt.detach().numpy(),
    #     vhat=vhat.detach().numpy(),
    #     corr_arr_v_vhat=corr_arr_v_vhat,
    # )
