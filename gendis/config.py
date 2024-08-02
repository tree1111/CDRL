from typing import Optional, Union

import yaml
from pydantic import BaseModel


class TrainConfig(BaseModel):
    max_epochs: int
    auto_lr_find: Union[bool, int]
    gpus: int


class VAEConfig(BaseModel):
    model_type: str
    hidden_size: int
    alpha: int
    dataset: str
    batch_size: Optional[int] = 64
    save_images: Optional[bool] = False
    lr: Optional[float] = None
    save_path: Optional[str] = None
    data_path: Optional[str] = None


class ConvVAEConfig(VAEConfig):
    channels: int
    height: int
    width: int


class LoggerConfig(BaseModel):
    name: str
    save_dir: str


class Config(BaseModel):
    model_configuration: Union[VAEConfig, ConvVAEConfig]
    train_config: TrainConfig
    model_type: str
    log_config: LoggerConfig


def load_config(path="config.yaml"):
    config = yaml.load(open(path), yaml.SafeLoader)
    model_type = config["model_params"]["model_type"]
    if model_type == "vae":
        model_config = VAEConfig(**config["model_params"])
    elif model_type == "conv-vae":
        model_config = ConvVAEConfig(**config["model_params"])
    else:
        raise NotImplementedError(f"Model {model_type} is not implemented")
    train_config = TrainConfig(**config["training_params"])
    log_config = LoggerConfig(**config["logger_params"])
    config = Config(
        model_configuration=model_config,
        train_config=train_config,
        model_type=model_type,
        log_config=log_config,
    )

    return config


config = load_config()
