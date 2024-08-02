from torchvision import datasets, transforms
from gendis.datasets import MultiDistrDataModule

from pytorch_lightning import Trainer
from models import vae_models
from config import config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from torchvision.utils import save_image


def make_model(config):
    model_type = config.model_type
    model_config = config.model_config

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict())


def load_vae_model(config, data_root, model_root):
    config.model_config.data_path = data_root
    model = make_model(config)
    checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def make_vae_reduction_dataset(model, data_root, new_root):
    '''
    :param model: vae model
    :param data_root: old data root
    :param new_data_root: new data root. please notice your data file will have the same file name as the original data,
            so please find new data under the new_data_root folder. it should be under new_data_root/CausalBarMNIST/chain/
    '''

    assert data_root != new_root

    # please do not change this transform sincethe vae model used this data_transform to train.
    # This is only for vae model.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1.)])

    # please do not set
    data_module = MultiDistrDataModule(
        root=data_root,
        graph_type='chain',
        batch_size=100,
        stratify_distrs=False,
        transform=data_transform,
        num_workers=1,
        train_size=1.0,
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

    if not os.path.exists(new_root):
        os.makedirs(new_root)
    if not os.path.exists(new_root + "CausalBarMNIST/"):
        os.makedirs(new_root + "CausalBarMNIST/")
    if not os.path.exists(new_root + "CausalBarMNIST/" + "chain/"):
        os.makedirs(new_root + "CausalBarMNIST/" + "chain/")

    imgs_fname = new_root + "CausalBarMNIST/" + "chain/" + "chain-imgs-train.pt"
    labels_fname = new_root + "CausalBarMNIST/" + "chain/" + "chain-labels-train.pt"
    targets_fname = new_root + "CausalBarMNIST/" + "chain/" + "chain-targets-train.pt"

    torch.save(all_x, imgs_fname)
    torch.save(all_labels, labels_fname)
    torch.save(all_targets, targets_fname)


if __name__ == "__main__":
    # your original data root
    data_root = '../../dat'

    # your model root, you do not need this d. you can just write: model_root = '...'
    d = config.model_config.dict()['save_path'] + '/' 'zdim_' + str(config.model_config.dict()['hidden_size']) \
        + '_alpha' + str(config.model_config.dict()['alpha'])
    model_root = d + f"/saved_models/{config.model_type}_alpha_{config.model_config.alpha}" \
                     f"_dim_{config.model_config.hidden_size}.ckpt"

    # load vae model
    model = load_vae_model(config, data_root, model_root)
    model.eval()
    # set new data root, must be different with the data_root
    new_root = 'causalbar_reduction_dat/'

    # make new data
    make_vae_reduction_dataset(model, data_root, new_root)

    # demo to load dataloader. please make sure transform is None. d
    new_data_module = MultiDistrDataModule(
        root=new_root,
        graph_type='chain',
        batch_size=32,
        stratify_distrs=True,
        transform=None,
        num_workers=1,
    )
    new_data_module.setup()

    dataiter = iter(new_data_module.train_dataloader())

    x, labels, targets = next(dataiter)
    print(x.shape, labels.shape, targets.shape)
    with torch.no_grad():
        images = model.decoder(x)
    save_image(
        images,
        "samples.png",
    )

    print(labels)


