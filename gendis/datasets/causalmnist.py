import collections
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from scipy import stats
from torchvision.datasets.mnist import MNIST
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import pil_to_tensor

from .morphomnist import morpho, perturb


def latent_scm_single_digit_params(graph, label, n_samples, intervention_idx=None):
    """Latent SCM for a single digit.

    Parameters
    ----------
    graph : str
        The type of latent causal graph we consider.
    label : int
        The label we consider.
    n_samples : int
        The number of samples.

    Returns
    -------
    meta_labels : dict of lists of length (n_samples,)
        Contains the meta-labels for the dataset.
        - width: the width of the digit
        - color: the color of the digit on the viridis colormap.
        - fracture_thickness: the thickness of the fracture
        - fracture_num_fractures: the number of fractures
    """
    meta_labels = dict()

    if graph == "chain":
        # the width of the digit is correlated with the color
        # which is correlated with the presence of a fracture
        if intervention_idx == 1:
            width_intervention = stats.skewnorm.rvs(a=0, loc=2, size=(n_samples, 1))
            width_intervention = width_intervention - min(
                width_intervention
            )  # Shift the set so the minimum value is equal to zero.
        else:
            width_intervention = 1.0

        if intervention_idx == 2:
            color_intervention = -4.0
        else:
            color_intervention = 1.0

        if intervention_idx == 3:
            color_intervention = 0.01
        else:
            color_intervention = 1.0
        # ensures that the width is at least 2
        width = (torch.rand(size=(n_samples, 1)) * 2) * width_intervention
        width = torch.minimum(width, torch.FloatTensor([2.0]).expand_as(width))

        # different colors correlate with the fracture
        # - thickness of the fracture will be more
        # - number of fractures for darker colors
        thickness = stats.skewnorm.rvs(a=2 * width, loc=1, size=(n_samples, 1))
        thickness = ((thickness - min(thickness)) / max(thickness) + 1) * 5
        thickness = torch.Tensor(thickness)
        num_fractures = torch.bernoulli(width / max(width)) + torch.bernoulli(width / max(width))

        # the thicker the fracture(s), the higher up color it is
        # on the spectrum
        color_value = stats.skewnorm.rvs(
            a=-thickness * num_fractures / 2.0 * color_intervention, loc=1, size=(n_samples, 1)
        )
        color_value = color_value - min(color_value)
        color_value = color_value / max(color_value)
        color_value = torch.Tensor(color_value)
        print("Color value: ", color_value.shape)

        meta_labels["width"] = width.to(torch.float32)
        meta_labels["color"] = color_value.to(torch.float32)
        meta_labels["fracture_thickness"] = thickness.to(torch.float32)
        meta_labels["fracture_num_fractures"] = num_fractures.to(torch.float32)
    elif graph == "collider":
        # the width of the digit is causes both the color
        # and the presence of a fracture
        width = torch.rand(size=(n_samples, 1)) * 2

        # the thicker the digit, the higher up color it is
        # on the spectrum
        color_value = stats.skewnorm.rvs(a=-width, loc=1, size=(n_samples, 1))
        color_value = color_value - min(color_value)
        color_value = color_value / max(color_value)
        color_value = torch.Tensor(color_value)
        print("Color value: ", color_value.shape)

        # different colors correlate with the fracture
        # - thickness of the fracture will be more
        # - number of fractures for darker colors
        # different colors correlate with the fracture
        # - thickness of the fracture will be more
        # - number of fractures for darker colors
        thickness = stats.skewnorm.rvs(a=2 * width, loc=1, size=(n_samples, 1))
        thickness = ((thickness - min(thickness)) / max(thickness) + 1) * 5
        thickness = torch.Tensor(thickness)
        num_fractures = torch.bernoulli(width) + torch.bernoulli(width)

        meta_labels["width"] = width.to(torch.float32)
        meta_labels["color"] = color_value.to(torch.float32)
        meta_labels["fracture_thickness"] = thickness.to(torch.float32)
        meta_labels["fracture_num_fractures"] = num_fractures.to(torch.float32)

    meta_labels["label"] = [label] * n_samples
    # add intervention targets
    if intervention_idx is None or intervention_idx == 0:
        meta_labels["intervention_targets"] = [[0, 0, 0]] * n_samples
    elif intervention_idx == 1:
        # intervene and change the distribution of the width
        meta_labels["intervention_targets"] = [[1, 0, 0]] * n_samples
    elif intervention_idx in [2, 3]:
        meta_labels["intervention_targets"] = [[0, 0, 1]] * n_samples
    return meta_labels


def get_mnist_digit(images, labels, target):
    raw_images = []
    for image, label in zip(images, labels):
        if label == target:
            raw_images.append(image)
    return raw_images


def apply_perturbation(image, perturbation, convert_dtype=True):
    # Convert image to binary
    image = image.squeeze().numpy()
    binary_image = np.array(image) > 0

    morph = morpho.ImageMorphology(binary_image)
    perturbed_image = perturbation(morph)
    if convert_dtype:
        return Image.fromarray((perturbed_image * 255).astype(np.uint8))
    else:
        return perturbed_image * 255


def _prepare_image(raw_imgs, meta_labels, idx, cmap, label):
    # apply changes to the image
    width, color, fracture_thickness, num_fractures = (
        meta_labels["width"][idx],
        meta_labels["color"][idx],
        meta_labels["fracture_thickness"][idx],
        meta_labels["fracture_num_fractures"][idx],
    )
    dtype = "uint8"
    # apply width changes
    if width < 1.0:
        width_func = perturb.Thinning(amount=width)
    else:
        width_func = perturb.Thickening(amount=width)
    fracture = perturb.Fracture(thickness=fracture_thickness, num_frac=int(num_fractures))

    img = raw_imgs[idx]
    img = img.squeeze()
    for perturb_func in [fracture, width_func]:
        img = torch.Tensor(np.array(img))
        img = apply_perturbation(img, perturb_func, convert_dtype=False)

    # change the color
    arr = img
    h, w = arr.shape
    color_value = cmap(color)[:3].squeeze()
    colored_arr = np.zeros((h, w, 3), dtype=dtype)
    mask = arr > 0  # Mask to identify the digit
    for i in range(3):
        colored_arr[:, :, i][mask] = color_value[i] * 255  # Apply uniform color to the digit

    # convert to PIL image
    img = Image.fromarray(colored_arr)

    # create dictionary of the meta label
    meta_label = dict()
    meta_label["width"] = meta_labels["width"][idx]
    meta_label["color"] = meta_labels["color"][idx]
    meta_label["fracture_thickness"] = meta_labels["fracture_thickness"][idx]
    meta_label["fracture_num_fractures"] = meta_labels["fracture_num_fractures"][idx]
    meta_label["label"] = label
    meta_label["intervention_targets"] = meta_labels["intervention_targets"][idx]
    return (img, meta_label)


# Define the dataset loader with perturbations
class CausalMNIST(MNIST):
    def __init__(
        self,
        root,
        graph_type,
        label=0,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        n_jobs=None,
        intervention_idx=None,
    ):
        super(CausalMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.intervention_idx = intervention_idx
        self.label = label
        self.n_jobs = n_jobs
        self.graph_type = graph_type

    def prepare_dataset(self, overwrite=False):
        """Prepare the dataset by applying the perturbations."""
        dataset_path = Path(self.root) / self.__class__.__name__ / self.graph_type
        
        # defaults to the observational case only
        intervention_idx_list = self.intervention_idx if self.intervention_idx is not None else [0]
        for idx, intervention_idx in enumerate(intervention_idx_list):
            dataset_path = Path(self.root) / self.__class__.__name__ / self.graph_type
    
            # None is a misnomer for intervention_idx being 0
            if intervention_idx is None:
                intervention_idx = 0

            # check if the dataset already exists, and if so, just load it
            if self.train:
                dataset_fpath = dataset_path / f"{self.graph_type}-{intervention_idx}-train.pt"
            else:
                dataset_fpath = dataset_path / f"{self.graph_type}-{intervention_idx}-test.pt"

            if dataset_fpath.exists() and not overwrite:
                continue

            if self.train:
                mnist_data = MNIST(self.root, train=True, download=True)
            else:
                mnist_data = MNIST(self.root, train=False, download=True)

            raw_imgs = get_mnist_digit(mnist_data.data, mnist_data.targets, self.label)
            n_samples = len(raw_imgs)

            print("\n\nGenerating dataset: for label", self.label, "with", n_samples, "samples")

            meta_labels = latent_scm_single_digit_params(
                self.graph_type,
                self.label,
                n_samples=n_samples,
                intervention_idx=intervention_idx,
            )
            cmap = plt.cm.viridis  # Use a continuous colormap like viridis and extract RGB values

            # now actually generate the data per image
            dataset = Parallel(n_jobs=self.n_jobs)(
                delayed(_prepare_image)(raw_imgs, meta_labels, idx, cmap, self.label)
                for idx in range(n_samples)
            )
            # imgs = torch.zeros((n_samples, 3, 28, 28))
            imgs = []
            meta_labels = collections.defaultdict(list)
            for idx, (img, meta_label) in enumerate(dataset):
                imgs.append(img)
                for key in meta_label.keys():
                    meta_labels[key].append(meta_label[key])

            dataset_path.mkdir(exist_ok=True, parents=True)
            torch.save(dataset, dataset_fpath)
        
        # actually load the dataset
        for idx, intervention_idx in enumerate(intervention_idx_list):
            dataset_path = Path(self.root) / self.__class__.__name__ / self.graph_type
    
            # None is a misnomer for intervention_idx being 0
            if intervention_idx is None:
                intervention_idx = 0

            # check if the dataset already exists, and if so, just load it
            if self.train:
                dataset_fpath = dataset_path / f"{self.graph_type}-{intervention_idx}-train.pt"
            else:
                dataset_fpath = dataset_path / f"{self.graph_type}-{intervention_idx}-test.pt"

            self._load_dataset(idx, dataset_fpath)
        self._prepare_metadata()

    def _load_dataset(self, idx, dataset_fpath):
        print(f'\n\nLoading dataset from "{dataset_fpath}"')
        dataset = torch.load(dataset_fpath)
        n_samples = len(dataset)

        imgs = []
        meta_labels = collections.defaultdict(list)
        for jdx, (img, meta_label) in enumerate(dataset):
            imgs.append(img)
            for key in meta_label.keys():
                meta_labels[key].append(meta_label[key])

        # add distribution indicator
        meta_labels['distribution_indicator'] = [idx] * n_samples

        if idx == 0:
            self.data = imgs
            self.meta_labels = meta_labels
            self.intervention_targets = torch.Tensor(self.meta_labels.get("intervention_targets"))
        else:
            self.data.extend(imgs)
            for key in meta_labels.keys():
                self.meta_labels[key].extend(meta_labels[key])
            # self.intervention_targets.extend(meta_labels["intervention_targets"])
            self.intervention_targets = torch.cat((self.intervention_targets, torch.Tensor(meta_labels["intervention_targets"])), dim=0)

    def _prepare_metadata(self):
        width = torch.tensor(self.meta_labels["width"])
        color = torch.tensor(self.meta_labels["color"])
        fracture_thickness = torch.tensor(self.meta_labels["fracture_thickness"])
        fracture_num_fractures = torch.tensor(self.meta_labels["fracture_num_fractures"])
        label = torch.tensor(self.meta_labels["label"])
        intervention_targets = torch.tensor(self.meta_labels["intervention_targets"])
        distr_indicators = torch.tensor(self.meta_labels["distribution_indicator"])
        # create Tensors for each dataset
        self.metadata = TensorDataset(
            width,
            color,
            fracture_thickness,
            fracture_num_fractures,
            label,
            distr_indicators,
            intervention_targets,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the image dataset.

        The target composes of the meta-labeling:
        - width
        - color
        - fracture_thickness
        - fracture_num_fractures
        - label
        """
        img, meta_label = self.data[index], self.metadata[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)

        return img, meta_label
