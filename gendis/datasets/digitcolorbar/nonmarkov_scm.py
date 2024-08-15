import torch

from .scm import truncated_normal


def nonmarkov_bar_digit_scm(intervention_idx, labels):
    """Generate parameters for the SCM of the MNIST dataset with a bar following a collider structure.

    Color-digit will affect the color-bar.

    Parameters
    ----------
    intervention_idx : int
        The intervention index.
    labels : tensor of shape (n_samples,)
        The labels of each of the MNIST digit samples. This corresponds to the
        digit.

    Returns
    -------
    causal_labels : dict
        A dictionary of:
            - digit : tensor of shape (n_samples,)
                The digit label.
            - color_digit : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
            - color_bar : tensor of shape (n_samples,)
                The value of the color-digit from [0, 1].
    digit_idx : list of int
        The indices of each image we want in the dataset. This may contain
        replicates of the indices of size (n_samples,).
    """
    n_samples = labels.shape[0]
    causal_labels = dict()

    # Digit is just uniformly distributed more or less in the MNIST dataset
    digit = torch.Tensor(labels)

    color_digit = torch.zeros(n_samples)
    color_bar = torch.zeros(n_samples)

    U_cdigit_cbar = torch.normal(mean=0.5, std=0.2, size=(n_samples, 1)).squeeze()
    color_digit = torch.zeros(n_samples)

    if intervention_idx == 0:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 1, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit
        causal_labels["intervention_targets"] = torch.Tensor([[0, 0, 0]] * n_samples)
    elif intervention_idx == 1:
        # Observational distribution
        # Color-digit will be a mixture of gaussians
        color_digit_means = torch.linspace(
            0, 2, 10
        )  # 10 possible digits, evenly spaced means from 0 to 1
        color_digit_means = torch.flip(color_digit_means, [0]) / color_digit_means.sum()
        color_digit_stds = 0.15 * torch.ones(10)  # Standard deviation of 0.15 for each digit
        causal_labels["intervention_targets"] = torch.Tensor([[0, 1, 0]] * n_samples)
    else:
        raise ValueError("Invalid intervention_idx. Must be 0, 1, 2 or 3.")

    # sample the color-digit conditioned on the digit
    for i in range(10):
        mask = digit == i
        num_samples = mask.sum().item()
        color_digit[mask] = truncated_normal(
            color_digit_means[i] + U_cdigit_cbar[mask], color_digit_stds[i], 0, 1, num_samples
        )

    # color-bar is caused by color-digit
    color_bar = truncated_normal(U_cdigit_cbar, 0.1, 0, 1, n_samples)

    causal_labels.update(
        {
            "digit": digit,
            "color_digit": color_digit,
            "color_bar": color_bar,
            "confounder": U_cdigit_cbar,
        }
    )

    return causal_labels
