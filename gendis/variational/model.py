import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image

from gendis.datasets import MultiDistrDataModule
from gendis.variational.models.conv_vae import Conv_VAE


def make_img_model(
    channels=1,
    height=28,
    width=28,
    lr=5e-3,
    lr_scheduler="cosine",
    hidden_size=32,
    alpha=1024,
    batch_size=144,
    save_images=False,
    save_path=None,
    confounded_vars=None,
):
    model = Conv_VAE(
        channels=channels,
        height=height,
        width=width,
        lr=lr,
        lr_scheduler=lr_scheduler,
        hidden_size=hidden_size,
        alpha=alpha,
        batch_size=batch_size,
        save_images=save_images,
        save_path=save_path,
    )
    return model


def make_vae_reduction_dataset(model, data_root, new_root, graph_type):
    """
    :param model: vae model
    :param data_root: old data root
    :param new_data_root: new data root. please notice your data file will have the same file name as the original data,
            so please find new data under the new_data_root folder. it should be under new_data_root/CausalBarMNIST/chain/
    """

    assert data_root != new_root

    # def transform_for_tanh(x):
    #     return 2 * x - 1.0

    # please do not change this transform sincethe vae model used this data_transform to train.
    # This is only for vae model.
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(transform_for_tanh)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # please do not set
    data_module = MultiDistrDataModule(
        root=data_root,
        graph_type=graph_type,
        batch_size=100,
        stratify_distrs=False,
        transform=data_transform,
        num_workers=1,
        train_size=1.0,
        dataset_name="digitcolorbar",
    )

    data_module.setup()

    all_x = []
    all_labels = []
    all_targets = []

    for images, labels, targets in data_module.train_dataloader():
        with torch.no_grad():
            x, _ = model.encode(images)

        all_x.extend(x)
        all_labels.extend(labels)
        all_targets.extend(targets)

    new_root = Path(new_root)
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    if not os.path.exists(new_root / "CausalDigitBarMNIST/"):
        os.makedirs(new_root / "CausalDigitBarMNIST/")
    if not os.path.exists(new_root / "CausalDigitBarMNIST/" / graph_type):
        os.makedirs(new_root / "CausalDigitBarMNIST/" / graph_type)

    imgs_fname = new_root / "CausalDigitBarMNIST/" / graph_type / f"{graph_type}-imgs-train.pt"
    labels_fname = new_root / "CausalDigitBarMNIST/" / graph_type / f"{graph_type}-labels-train.pt"
    targets_fname = (
        new_root / "CausalDigitBarMNIST/" / graph_type / f"{graph_type}-targets-train.pt"
    )
    print("Saving new dataset to: ", imgs_fname)
    torch.save(all_x, imgs_fname)
    torch.save(all_labels, labels_fname)
    torch.save(all_targets, targets_fname)


def test_img_model_from_chkpoint(
    checkpoint_fname,
    data_module,
):
    model = Conv_VAE.load_from_checkpoint(checkpoint_fname)

    # load in a sample batch from the dataset
    dataiter = iter(data_module.test_dataloader())

    x, labels, targets = next(dataiter)
    print(x.shape, labels.shape, targets.shape)
    with torch.no_grad():
        images = model.decoder(x)
    images = model.scale_image(images)
    save_image(
        images,
        "samples.png",
    )

    print(labels)

    return model


if __name__ == "__main__":
    data_root = Path("/Users/adam2392/pytorch_data/")
    graph_type = "nonmarkov"
    seed = 2

    graph_type = "collider"
    seed = 1

    # set new data root, must be different with the data_root
    new_root = data_root / "vae-reduction"

    # file path for the model checkpoint
    model_root = (
        data_root
        / "vae-reduction"
        / f"vae-reduction-cosinelr-batch32-{graph_type}-seed={seed}"
        # / "epoch=98-step=783090.ckpt"
        # / "epoch=1841-step=6216750.ckpt" # nonmarkov
        / "epoch=836-step=6620670.ckpt"  # collider
    )
    # load vae model
    model = Conv_VAE.load_from_checkpoint(model_root)

    # make new data
    print("Converting the dataset to embedding format")
    make_vae_reduction_dataset(model, data_root, new_root, graph_type)

    print("Loading the dataset...")
    # demo to load dataloader. please make sure transform is None. d
    new_data_module = MultiDistrDataModule(
        root=new_root,
        graph_type=graph_type,
        batch_size=16,
        stratify_distrs=True,
        transform=None,
        num_workers=1,
        dataset_name="digitcolorbar",
    )
    new_data_module.setup()

    dataiter = iter(new_data_module.test_dataloader())

    x, labels, targets = next(dataiter)
    print(x.shape, labels.shape, targets.shape)
    with torch.no_grad():
        images = model.decoder(x)
    images = model.scale_image(images)
    save_image(
        images,
        f"/Users/adam2392/Downloads/{graph_type}-test-samples.png",
    )

    print(labels)
