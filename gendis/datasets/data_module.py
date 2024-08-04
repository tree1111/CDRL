from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler, random_split

from . import CausalBarMNIST, CausalMNIST
from .digitcolorbar import CausalDigitBarMNIST
from .utils import summary_statistics


# Custom Stratified Sampler
class StratifiedSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.unique_labels = np.unique(labels)
        self.label_indices = {label: np.where(labels == label)[0] for label in self.unique_labels}
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        num_per_class = self.batch_size // len(self.unique_labels)

        for _ in range(self.num_samples // self.batch_size):
            batch_indices = []
            for label in self.unique_labels:
                label_indices = np.random.choice(
                    self.label_indices[label], num_per_class, replace=False
                )
                batch_indices.extend(label_indices)

            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)

        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class MultiDistrDataModule(LightningDataModule):
    """
    Data module for multi-distributional data.

    Attributes
    ----------
    medgp: MultiEnvDGP
        Multi-environment data generating process.
    num_samples_per_env: int
        Number of samples per environment.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the data loaders.
    intervention_targets_per_distr: Tensor, shape (num_envs, num_causal_variables)
        Intervention targets per environment, with 1 indicating that the variable is intervened on.
    log_dir: Optional[Path]
        Directory to save summary statistics and plots to. Default: None.
    intervention_target_misspec: bool
        Whether to misspecify the intervention targets. If true, the intervention targets are permuted.
        I.e. the model received the wrong intervention targets. Default: False.
    intervention_target_perm: Optional[list[int]]
        Permutation of the intervention targets. If None, a random permutation is used. Only used if
        intervention_target_misspec is True. Default: None.
    flatten: bool
        Whether to flatten the data. Default: False.

    Methods
    -------
    setup(stage=None) -> None
        Setup the data module. This is where the data is sampled.
    train_dataloader() -> DataLoader
        Return the training data loader.
    val_dataloader() -> DataLoader
        Return the validation data loader.
    test_dataloader() -> DataLoader
        Return the test data loader.
    """

    def __init__(
        self,
        root,
        graph_type,
        batch_size: int,
        stratify_distrs: bool = True,
        label: int = 0,
        num_workers: int = -1,
        train_size: float = 0.9,
        val_size: float = 0.05,
        transform=None,
        log_dir: Optional[Path] = None,
        dataset_name: str = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = Path(log_dir) if log_dir is not None else log_dir
        self.dataset_name = dataset_name

        self.stratify_distrs = stratify_distrs
        self.transform = transform
        self.train_size = train_size
        self.val_size = val_size

        self.root = root
        self.label = label
        self.graph_type = graph_type

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name == "digit":
            self.dataset = CausalDigitBarMNIST(
                root=self.root,
                graph_type=self.graph_type,
                train=True,
                n_jobs=None,
                transform=self.transform,
            )
        else:
            self.dataset = CausalBarMNIST(
                root=self.root,
                graph_type=self.graph_type,
                train=True,
                n_jobs=None,
                transform=self.transform,
            )

        train_size = int(self.train_size * len(self.dataset))
        val_size = int(self.val_size * (len(self.dataset) - train_size))
        test_size = len(self.dataset) - train_size - val_size

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(self.dataset, [train_size, val_size, test_size])

        if self.stratify_distrs:
            distr_labels = [x[1][-1] for x in self.train_dataset]
            self.train_sampler = StratifiedSampler(distr_labels, self.batch_size)

            distr_labels = [x[1][-1] for x in self.val_dataset]
            self.val_sampler = StratifiedSampler(distr_labels, self.batch_size)
        else:
            self.train_sampler = None
            self.val_sampler = None

    @property
    def meta_label_strs(self):
        return self.dataset.meta_label_strs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader


class ClusteredMultiDistrDataModule(LightningDataModule):
    """
    Data module for clustered multi-distributional data.

    Attributes
    ----------
    medgp: MultiEnvDGP
        Multi-environment data generating process.
    num_samples_per_env: int
        Number of samples per environment.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the data loaders.
    intervention_targets_per_distr: Tensor, shape (num_envs, num_causal_variables)
        Intervention targets per environment, with 1 indicating that the variable is intervened on.
    log_dir: Optional[Path]
        Directory to save summary statistics and plots to. Default: None.
    intervention_target_misspec: bool
        Whether to misspecify the intervention targets. If true, the intervention targets are permuted.
        I.e. the model received the wrong intervention targets. Default: False.
    intervention_target_perm: Optional[list[int]]
        Permutation of the intervention targets. If None, a random permutation is used. Only used if
        intervention_target_misspec is True. Default: None.
    flatten: bool
        Whether to flatten the data. Default: False.

    Methods
    -------
    setup(stage=None) -> None
        Setup the data module. This is where the data is sampled.
    train_dataloader() -> DataLoader
        Return the training data loader.
    val_dataloader() -> DataLoader
        Return the validation data loader.
    test_dataloader() -> DataLoader
        Return the test data loader.
    """

    def __init__(
        self,
        root,
        graph_type,
        batch_size: int,
        label: int = 0,
        intervention_types=None,
        num_workers: int = -1,
        train_size: float = 0.9,
        val_size: float = 0.05,
        transform=None,
        log_dir: Optional[Path] = None,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = Path(log_dir)

        self.transform = transform
        self.train_size = train_size
        self.val_size = val_size
        self.flatten = flatten

        self.root = root
        self.label = label
        self.graph_type = graph_type
        self.intervention_types = intervention_types

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = CausalMNIST(
            root=self.root,
            graph_type=self.graph_type,
            label=self.label,
            download=True,
            train=True,
            n_jobs=None,
            intervention_idx=self.intervention_types,
            transform=self.transform,
        )
        self.dataset.prepare_dataset()

        # meta_labels = collections.defaultdict(list)
        # distr_indicators = []

        # load dataset
        # datasets = []
        # intervention_targets_per_distr = []
        # hard_interventions_per_distr = None
        # num_distrs = 0
        # for intervention_idx in self.intervention_types:
        #     dataset = CausalMNIST(
        #         root=self.root,
        #         graph_type=self.graph_type,
        #         label=0,
        #         download=True,
        #         train=True,
        #         n_jobs=None,
        #         intervention_idx=intervention_idx,
        #         transform=self.transform,
        #     )
        #     dataset.prepare_dataset(overwrite=False)
        #     datasets.append(dataset)
        #     num_distrs += 1
        #     intervention_targets_per_distr.append(dataset.intervention_targets)

        # dataset = torch.utils.data.ConcatDataset(datasets)

        # load in all the pytorch datasets
        # for idx in range(len(self.datasets)):
        #     if idx == 0:
        #         x = self.datasets[idx].data
        #         meta_labels = deepcopy(self.datasets[idx].meta_labels)
        #     else:
        #         x = torch.cat([x, self.datasets[idx].data], dim=0)

        #         # update meta_labels
        #         for key in meta_labels.keys():
        #             meta_labels[key].extend(self.datasets[idx].meta_labels[key])
        #     distr_indicators.extend([idx] * len(self.datasets[idx]))
        # width = torch.tensor(meta_labels["width"])
        # color = torch.tensor(meta_labels["color"])
        # fracture_thickness = torch.tensor(meta_labels["fracture_thickness"])
        # fracture_num_fractures = torch.tensor(meta_labels["fracture_num_fractures"])
        # label = torch.tensor(meta_labels["label"])
        # intervention_targets = torch.tensor(meta_labels["intervention_targets"])
        # distr_indicators = torch.tensor(distr_indicators)

        # if self.flatten:
        #     # flatten the data per sample
        #     x = x.view(x.size(0), -1)

        # create Tensors for each dataset
        # dataset = TensorDataset(
        #     x,
        #     width,
        #     color,
        #     fracture_thickness,
        #     fracture_num_fractures,
        #     label,
        #     distr_indicators,
        #     intervention_targets,
        # )
        train_size = int(self.train_size * len(self.dataset))
        val_size = int(self.val_size * (len(self.dataset) - train_size))
        test_size = len(self.dataset) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(self.dataset, [train_size, val_size, test_size])

    def meta_label_strs(self):
        return [
            "width",
            "color",
            "fracture_thickness",
            "fracture_num_fractures",
            "label",
            "distr_indicators",
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
