from itertools import permutations

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from torch import Tensor
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef


def nonlinear_r2(v_hat: Tensor, v: Tensor, method: str = "linear"):
    if v_hat.shape != v.shape:
        raise RuntimeError(f"v_hat and v must have the same shape, got {v_hat.shape} and {v.shape}")

    if method == "linear":
        reg = LinearRegression(fit_intercept=True).fit(v_hat, v)
    elif method == "rf":
        reg = RandomForestRegressor().fit(v_hat, v)
    elif method == "svm":
        reg = SVR().fit(v_hat, v)
    elif method == "mlp":
        reg = MLPRegressor().fit(v_hat, v)

    v_pred = reg.predict(v_hat)
    r2s = r2_score(v, v_pred, multioutput="raw_values")
    corr_coefficients = np.mean(
        np.sqrt(r2s)
    )  # To be comparable to MCC (this is the average of R = coefficient of multiple correlation)
    print(v.shape, v_hat.shape, r2s.shape, corr_coefficients.shape)

    return corr_coefficients


def learned_vs_latents(v_hat: Tensor, v: Tensor, method="pearson"):
    """Compute the mean correlation coefficient between the columns of v_hat and v.

    Parameters
    ----------
    v_hat : Tensor of shape (n_samples, n_outputs)
        Estimated tensor.
    v : Tensor of shape (n_samples, n_outputs)
        Ground-truth tensor.
    method : str, optional
        Method for computing CCoeff, by default "pearson"

    Returns
    -------
    corr_coefficients : Tensor of shape (n_outputs,)
        The mean correlation coefficient between the columns of v_hat and v.
    """
    if v_hat.shape != v.shape:
        raise RuntimeError(f"v_hat and v must have the same shape, got {v_hat.shape} and {v.shape}")

    if method == "pearson":
        pearson = PearsonCorrCoef(num_outputs=v.shape[1])
        corr_coefficients = torch.abs(pearson(v_hat, v))
    elif method == "spearman":
        spearman = SpearmanCorrCoef(num_outputs=v.shape[1])
        corr_coefficients = torch.abs(spearman(v_hat, v))
    elif method == "linear":
        reg = LinearRegression(fit_intercept=True).fit(v_hat, v)
        v_pred = reg.predict(v_hat)
        r2s = r2_score(v, v_pred, multioutput="raw_values")
        corr_coefficients = np.mean(
            np.sqrt(r2s)
        )  # To be comparable to MCC (this is the average of R = coefficient of multiple correlation)
    return corr_coefficients


def mean_correlation_coefficient(v_hat: Tensor, v: Tensor, method="pearson") -> Tensor:
    """Compute the mean correlation coefficient between the columns of v_hat and v.

    Parameters
    ----------
    v_hat : Tensor of shape (n_samples, n_outputs)
        Estimated tensor.
    v : Tensor of shape (n_samples, n_outputs)
        Ground-truth tensor.
    method : str, optional
        Method for computing CCoeff, by default "pearson"

    Returns
    -------
    corr_coefficients : Tensor of shape (n_outputs,)
        The mean correlation coefficient between the columns of v_hat and v.
    """
    assert v_hat.shape == v.shape

    if method == "pearson":
        # get all permutations of the columns of v_hat and v
        v_hat_permutations = list(permutations(range(v_hat.shape[1])))
        corr_coefficients_perm = torch.zeros(
            (len(v_hat_permutations), v_hat.shape[1]), device=v_hat.device
        )
        for p_idx, perm in enumerate(permutations(range(v_hat.shape[1]))):
            for i in range(v_hat.shape[1]):
                data = torch.stack([v_hat[:, i], v[:, perm[i]]], dim=1).T
                corr_coefficients_perm[p_idx, i] = torch.abs(torch.corrcoef(data)[0, 1])
        best_p_idx = torch.argmax(corr_coefficients_perm.sum(dim=1))
        corr_coefficients = corr_coefficients_perm[best_p_idx]
    elif method == "spearman":
        # Note: spearman does not check all permutations and does not work with
        # intervention target permutation
        spearman = SpearmanCorrCoef(num_outputs=v.shape[1])
        corr_coefficients = torch.abs(spearman(v_hat, v))
    else:
        raise ValueError(f"Unknown correlation coefficient method: {method}")
    return corr_coefficients
