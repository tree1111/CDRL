import torch
import numpy as np
from scipy.stats import truncnorm

# Number of samples
n_samples = 1000


def truncated_normal(mean, std, lower, upper, size):
    return truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs(size)


def bar_scm(n_samples, intervention_idx, label):
    meta_labels = dict()

    # Truncated normal distribution parameters
    mu_cd, sigma_cd = 0, 0.2
    mu_cb, sigma_cb = 0, 0.2
    width_lim = [-2, 2]
    width = np.random.uniform(*width_lim, n_samples)

    if intervention_idx == 0:
        mu_cb, sigma_cb = 0, 0.2
    elif intervention_idx == 1:
        mu_cb, sigma_cb = 0.3, 0.2
    elif intervention_idx == 2:
        mu_cb, sigma_cb = -0.3, 0.2
    elif intervention_idx == 3:
        width_lim = [-1, 1]
        width = truncated_normal(0, 1, *width_lim, n_samples)

    # sample exogenous noise for color of digit and color of bar
    noise_cd = truncated_normal(mu_cd, sigma_cd, -0.5, 0.5, n_samples)
    noise_cb = truncated_normal(mu_cb, sigma_cb, -0.5, 0.5, n_samples)

    # Generate samples for observational setting
    color_bar = np.clip((width + 2) / 4 + noise_cb, 0, 1)
    color_digit = np.clip(color_bar + noise_cd, 0, 1)

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
    return meta_labels, width, color_bar, color_digit
