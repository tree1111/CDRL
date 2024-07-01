from pathlib import Path
import numpy as np
import torch
from scipy import stats
from PIL import Image
from joblib import delayed, Parallel
from .morphomnist import morpho, perturb

from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt


def latent_scm_single_digit_params(graph, label, n_samples):
    """_summary_

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
    _type_
        _description_
    """
    meta_labels = dict()

    if graph == "chain":
        # the width of the digit is correlated with the color
        # which is correlated with the presence of a fracture
        #
        width = torch.rand(size=(n_samples, 1)) * 2

        # the thicker the digit, the higher up color it is
        # on the spectrum
        color_value = stats.skewnorm.rvs(a=-width,loc=1, size=(n_samples,1))
        color_value = (color_value - min(color_value))
        color_value = color_value / max(color_value)
        color_value = torch.Tensor(color_value)
        print("Color value: ", color_value.shape)

        # different colors correlate with the fracture
        # - thickness of the fracture will be more
        # - number of fractures for darker colors
        thickness = stats.skewnorm.rvs(a=2*color_value,loc=1, size=(n_samples,1))
        thickness = ((thickness - min(thickness)) / max(thickness) + 1) * 5
        thickness = torch.Tensor(thickness)
        num_fractures = torch.bernoulli(color_value) + torch.bernoulli(color_value)
        fracture_params = (thickness, num_fractures)

        meta_labels["width"] = width
        meta_labels["color"] = color_value
        meta_labels["fracture_thickness"] = thickness
        meta_labels["fracture_num_fractures"] = num_fractures
    elif graph == "collider":
        # the width of the digit is causes both the color
        # and the presence of a fracture
        width = torch.rand(size=(n_samples, 1)) * 2

        # the thicker the digit, the higher up color it is
        # on the spectrum
        color_value = torch.rand(width)

        # different colors correlate with the fracture
        # - thickness of the fracture will be more
        # - number of fractures for darker colors
        thickness = torch.rand()
        num_fractures = torch.bernoulli(width) + torch.bernoulli(width)
        fracture_params = (thickness, num_fractures)

        meta_labels["width"] = width
        meta_labels["color"] = color_value
        meta_labels["fracture_thickness"] = thickness
        meta_labels["fracture_num_fractures"] = num_fractures

    meta_labels["label"] = [label] * n_samples
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

def _prepare_image(dataset, meta_labels, idx, cmap, label):
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

    img = dataset.data[idx]
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
    return (img, meta_label)


# Define the dataset loader with perturbations
class CausalMNIST(MNIST):
    def __init__(
        self,
        root,
        graph_type,
        label="0",
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        n_jobs=None,
    ):
        super(CausalMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.label = label
        self.n_jobs = n_jobs
        self.graph_type = graph_type

    def prepare_dataset(self):
        dataset_path = Path(self.root) / self.__class__.__name__ / self.graph_type
        if dataset_path.exists():
            return

        if self.train:
            mnist_data = MNIST(self.root, train=True, download=True)
        else:
            mnist_data = MNIST(self.root, train=False, download=True)

        n_samples = len(mnist_data)
        meta_labels = latent_scm_single_digit_params(
            self.graph_type,
            self.label,
            n_samples=n_samples,
        )
        cmap = plt.cm.viridis  # Use a continuous colormap like viridis and extract RGB values

        # now actually generate the data per image
        dataset = Parallel(n_jobs=self.n_jobs)(
            delayed(_prepare_image)(mnist_data, meta_labels, idx, cmap, self.label) for idx in range(n_samples)
        )

        dataset_path.mkdir(exist_ok=True, parents=True)
        torch.save(dataset, dataset_path / f"{self.graph_type}.pt")

    def __getitem__(self, index):
        """Get a sample from the image dataset.

        The target composes of the meta-labeling:
        - width
        - color
        - fracture_thickness
        - fracture_num_fractures
        - label
        """
        img, target = super(CausalMNIST, self).__getitem__(index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
