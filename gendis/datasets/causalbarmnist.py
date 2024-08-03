from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from gendis.datasets.morphomnist import morpho, perturb


def truncated_normal(mean, std, lower, upper, size):
    return truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs(size)


# Function to uniformly distribute clipped values of color_digit
def post_process_color_digit(color_digit, bad_val=0):
    zero_indices = color_digit == bad_val
    color_digit[zero_indices] = np.random.uniform(0, 1, zero_indices.sum())
    return color_digit


def add_bar(img, color_bar_val=255, start_height=0, height=3, start_width=0, width=None):
    if width is None:
        width = img.shape[1]

    # add bar
    for color_idx in range(3):
        img[
            start_height : start_height + height,
            start_width : start_width + width,
            color_idx,
        ] = (
            color_bar_val[color_idx] * 255.0
        )
    return img


def value_to_rgb(value, methods=None, cmap=None):
    # Map to RGB components
    if methods == "cmap":
        color_value = np.array(cmap(value)[:3]).squeeze()
    elif methods == "interpolate":
        start = np.array([1, 1, 0])
        end = np.array([0, 0, 1])
        color_value = start + value * (end - start)
    else:
        r = value
        g = 1 - value
        b = value + (1 - value) / 2
        color_value = np.array([r, g, b])
    return color_value


def apply_perturbation(image, perturbation, convert_dtype=True):
    # Convert image to binary
    image = image.squeeze().numpy()
    binary_image = np.array(image) > 0

    morph = morpho.ImageMorphology(
        binary_image,
        # threshold=0.4, scale=4
    )
    perturbed_image = perturbation(morph)

    # perturbed_image = morph.downscale(perturbed_image)
    if convert_dtype:
        return Image.fromarray((perturbed_image * 255).astype(np.uint8))
    else:
        return perturbed_image * 255


def bar_scm(n_samples, intervention_idx, label):
    meta_labels = dict()

    # Truncated normal distribution parameters
    mu_cd, sigma_cd = 0, 0.2
    mu_cb, sigma_cb = 0, 0.2
    width_lim = [-0.95, 0.95]
    width = np.random.uniform(*width_lim, n_samples)

    if intervention_idx == 0:
        mu_cb, sigma_cb = 0, 0.2
    elif intervention_idx == 1:
        mu_cd, sigma_cd = 0.6, 0.2
    elif intervention_idx == 2:
        mu_cd, sigma_cd = 0.9, 0.05
    elif intervention_idx == 3:
        # width_lim = [-0.5, 0.5]
        width_lim = [-0.95, 0.95]
        width = truncated_normal(0.5, 0.5, *width_lim, n_samples)

    # sample exogenous noise for color of digit and color of bar
    noise_cd = truncated_normal(mu_cd, sigma_cd, 0.0, 1.0, n_samples)
    # noise_cb = truncated_normal(mu_cb, sigma_cb, 0, 1, n_samples)

    color_bar = truncated_normal(np.mean(width), 0.05, 0, 1, n_samples)
    color_bar = post_process_color_digit(color_bar, bad_val=0)

    color_digit = np.clip(color_bar + noise_cd, 0, 1)
    color_digit = post_process_color_digit(color_digit, bad_val=0)
    # Generate samples for observational setting
    # color_bar = np.clip((width + 2) / 4 + noise_cb, 0, 1)
    # color_digit = np.clip(color_bar + noise_cd, 0, 1)

    width = torch.Tensor(width)
    color_bar = torch.Tensor(color_bar)
    color_digit = torch.Tensor(color_digit)

    meta_labels = dict()
    meta_labels["width"] = width.to(torch.float32)
    meta_labels["color_digit"] = color_digit.to(torch.float32)
    meta_labels["color_bar"] = color_bar.to(torch.float32)
    meta_labels["label"] = [label] * n_samples
    # add intervention targets
    if intervention_idx is None or intervention_idx == 0:
        meta_labels["intervention_targets"] = [[0, 0, 0]] * n_samples
    elif intervention_idx in [1, 2]:
        # intervene and change the distribution of the color-digit
        meta_labels["intervention_targets"] = [[0, 0, 1]] * n_samples
    elif intervention_idx in [3]:
        # intervene and change the distribution of the digit width
        meta_labels["intervention_targets"] = [[1, 0, 0]] * n_samples

    # convert to a list of dictionaries, where each list element corresponds
    # to the meta labels of a single sample
    meta_labels = [
        {key: meta_labels[key][i] for key in meta_labels.keys()} for i in range(n_samples)
    ]
    return meta_labels, width, color_bar, color_digit


# Placeholder for alter_img function
def alter_img(img, width, color_digit, color_bar):
    cmap = plt.cm.viridis

    # Apply transformations to the image based on the parameters
    # This is just a placeholder, you need to implement the actual transformation logic
    # apply width changes
    if width < 0.0:
        width = np.abs(width)
        width_func = perturb.Thinning(amount=width)
    else:
        width_func = perturb.Thickening(amount=width)

    img = img.squeeze()
    for perturb_func in [width_func]:
        img = torch.Tensor(np.array(img))
        img = apply_perturbation(img, perturb_func, convert_dtype=False)

    # change the color
    h, w = img.shape
    # print(color_digit, color_bar)
    # color_value = np.array(cmap(color_digit.item())[:3]).squeeze()
    color_value = value_to_rgb(color_digit.item())
    colored_arr = np.zeros((h, w, 3), dtype=np.uint8)
    mask = img > 0  # Mask to identify the digit
    for i in range(3):
        colored_arr[:, :, i][mask] = color_value[i] * 255  # Apply uniform color to the digit
    img = colored_arr

    # add bar
    # color_value = np.array(cmap(color_digit.item())[:3]).squeeze()
    color_value = value_to_rgb(color_digit.item())
    img = add_bar(
        img,
        color_bar_val=color_value,
        start_height=0,
        height=4,
        start_width=0,
        width=None,
    )
    # print('Finished altering image: ', img.shape)
    return img


# Function to create a new dataset
def create_altered_mnist_dataset(
    mnist_loader: DataLoader, label, n_samples=1000, intervention_idx=0, n_jobs=None
):
    # images are stored in a list of (3, 28, 28) arrays
    altered_images = []

    # labels are stored as a list of dictionaries
    labels = []

    meta_labels, width, color_bar, color_digit = bar_scm(n_samples, intervention_idx, label=label)

    if n_jobs is None:
        n_jobs = 1

    def process_batch(img, meta_label, width, color_digit, color_bar):
        altered_img = alter_img(img, width, color_digit, color_bar)
        return altered_img, meta_label

    idx = 0
    results = []
    while idx < n_samples:
        for img, _ in mnist_loader:
            if idx >= n_samples:
                break
            # Add to results list for parallel processing
            meta_label = meta_labels[idx]
            results.append(
                delayed(process_batch)(
                    img, meta_label, width[idx], color_digit[idx], color_bar[idx]
                )
            )
            idx += 1

    # Execute parallel processing
    processed_results = Parallel(n_jobs=n_jobs)(results)

    # Collect results
    for altered_img, label in processed_results:
        altered_images.append(Image.fromarray(altered_img, mode="RGB"))
        # altered_images.append(torch.Tensor(np.transpose(altered_img, (2, 0, 1))))
        labels.append(label)
    # else:
    #     idx = 0
    #     while idx < n_samples:
    #         for img, _ in mnist_loader:
    #             if idx >= n_samples:
    #                 break
    #             altered_img = alter_img(img, width[idx], color_digit[idx], color_bar[idx])
    #             altered_images.append(altered_img)

    #             meta_label = meta_labels[idx]
    #             labels.append(meta_label)
    #             idx += 1

    return altered_images, labels


# Custom dataset to filter images of digit 0
class SingleDigitDataset(Dataset):
    def __init__(self, mnist_dataset, digit):
        self.data = [img for img, label in mnist_dataset if label == digit]
        self.data = np.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


# Define the dataset loader with perturbations
class CausalBarMNIST(Dataset):
    def __init__(
        self,
        root,
        graph_type,
        train=True,
        transform=None,
        target_transform=None,
        n_jobs=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.n_jobs = n_jobs
        self.graph_type = graph_type

        root = Path(root)

        # load data from disc
        self.data = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-imgs-train.pt"
        )
        self.labels = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-labels-train.pt"
        )
        if isinstance(self.labels, list):
            self.labels = torch.vstack(self.labels)

        self.intervention_targets = torch.load(
            root / self.__class__.__name__ / graph_type / f"{graph_type}-targets-train.pt"
        )
        if isinstance(self.intervention_targets, list):
            self.intervention_targets = torch.vstack(self.intervention_targets)

        if not all(
            [
                len(self.data) == len(self.labels),
                len(self.data) == len(self.intervention_targets),
            ]
        ):
            raise ValueError("Data, labels and intervention targets must have the same length.")

    @property
    def intervention_targets_per_distr(self):
        return [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]

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
        img, meta_label, target = (
            self.data[index],
            self.labels[index],
            self.intervention_targets[index],
        )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)

        return img, meta_label, target

    @property
    def width(self):
        return self.labels[:, 0]

    @property
    def width_idx(self):
        return 0

    @property
    def color_bar_idx(self):
        return 2
    
    @property
    def color_digit_idx(self):
        return 1

    @property
    def color_bar(self):
        return self.labels[:, 2]

    @property
    def color_digit(self):
        return self.labels[:, 1]

    @property
    def latent_dim(self):
        return self.labels.shape[1]

    @property
    def distribution_idx(self):
        return self.labels[:, 4]


if __name__ == "__main__":
    root = Path("/Users/adam2392/pytorch_data/")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,  # transform=transform
    )
    digit = 0
    n_jobs = 4

    digit_zero_dataset = SingleDigitDataset(mnist_dataset, digit=digit)
    mnist_loader = DataLoader(digit_zero_dataset, batch_size=1, shuffle=True)

    # # create the altered dataset
    # imgs, labels = create_altered_mnist_dataset(
    #     mnist_loader, label=digit, n_samples=10, intervention_idx=0, n_jobs=n_jobs
    # )
    # print(imgs.shape)
    # print(len(labels))

    all_imgs = []
    all_labels = []
    n_samples = 10

    for intervention_idx in [0, 1, 2, 3]:
        imgs, labels = create_altered_mnist_dataset(
            mnist_loader,
            label=digit,
            n_samples=n_samples,
            intervention_idx=intervention_idx,
            n_jobs=n_jobs,
        )

        all_imgs.extend(imgs)

        for label in labels:
            label["distr_idx"] = intervention_idx
        all_labels.extend(labels)

    # now save the latent factors and the intervention targets
    # per image
    keys = ["width", "color_digit", "color_bar", "label", "distr_idx"]
    label_tensor = torch.zeros((len(all_labels), len(keys)))
    # convert the labels from a list of dictionaries to a tensor array
    for idx, key in enumerate(keys):
        label_tensor[:, idx] = torch.Tensor([label[key] for label in all_labels])

    intervention_target_tensor = torch.zeros((len(all_labels), 3), dtype=torch.int)
    intervention_target_tensor[:] = torch.Tensor(
        [label["intervention_targets"] for label in all_labels]
    )

    # save the actual data to disc now
    imgs_fname = root / "CausalBarMNIST" / "chain" / "chain-imgs-train.pt"
    labels_fname = root / "CausalBarMNIST" / "chain" / "chain-labels-train.pt"
    targets_fname = root / "CausalBarMNIST" / "chain" / "chain-targets-train.pt"
    imgs_fname.parent.mkdir(exist_ok=True, parents=True)

    torch.save(all_imgs, imgs_fname)
    torch.save(label_tensor, labels_fname)
    torch.save(intervention_target_tensor, targets_fname)
