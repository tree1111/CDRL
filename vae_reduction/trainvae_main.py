from pytorch_lightning import Trainer
from models import vae_models
from config.config import load_config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os


def make_model(config):
    model_type = config.model_type
    model_config = config.model_configuration

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict())


if __name__ == "__main__":
    # please modify train_config.yaml for hyperparameters: data path, save path,
    config = load_config(path="train_config.yaml")
    d = config.model_configuration.dict()['save_path'] + '/' 'zdim_' + str(config.model_configuration.dict()['hidden_size']) \
        + '_alpha' + str(config.model_configuration.dict()['alpha'])
    if not os.path.exists(d):
        os.makedirs(d)
    config.model_configuration.save_path = d + '/log_img'
    config.log_config.save_dir = d + '/log'
    model = make_model(config)
    train_config = config.train_config

    logger = TensorBoardLogger(**config.log_config.dict())
    trainer = Trainer(**train_config.dict(), logger=logger,
                      callbacks=LearningRateMonitor(), accelerator='gpu', devices=1)
    if train_config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        new_lr = lr_finder.suggestion()
        print("Learning Rate Chosen:", new_lr)
        model.lr = new_lr
        trainer.fit(model)
    else:
        trainer.fit(model)

    if not os.path.isdir("./saved_models"):
        os.mkdir("./saved_models")
    trainer.save_checkpoint(
        f"saved_models/{config.model_type}_alpha_{config.model_configuration.alpha}_dim_{config.model_configuration.hidden_size}.ckpt")
