import normflows as nf

import networkx as nx
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Uniform
from torch.nn import ParameterList


def make_spline_flows(
    n_flows: int,
    latent_dim: int,
    n_hidden_dim: int,
    n_layers: int,
    permutation: bool = True,
) -> list[nf.flows.Flow]:
    """_summary_

    Parameters
    ----------
    n_flows : int
        The number of flows to generate
    latent_dim : int
        The dimensionality of the latent space.
    n_hidden_dim : int
        The dimensionality of the neural network parametrization of
        the splines.
    n_layers : int
        Number of layers to use to parametrize the neural spline.
    permutation : bool, optional
        Whether to add a permutation of the flow, by default True.

    Returns
    -------
    list[nf.flows.Flow]
        List of normalizing flows.
    """
    flows = []
    for i in range(n_flows):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(latent_dim, n_layers, n_hidden_dim)
        ]
        if permutation:
            flows += [nf.flows.LULinearPermute(latent_dim)]
    return flows


def set_initial_edge_coeffs(
    dag: nx.DiGraph,
    min_val: float = -1,
    max_val: float = 1,
    device: torch.device = None,
) -> tuple[list[ParameterList], list[list[bool]]]:
    """Set initial coefficient values for a linear SCM and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    min_val : float
        The minimum value for the coefficient. Default is -1.
    max_val : float
        The maximum value for the coefficient. Default is 1.
    device : torch.device
        PyTorch device. Default is None, and will be inferred.

    Returns
    -------
    coeff_values : list[ParameterList]
        The coefficient values for each node in the SCM. Each element has a list
        of coefficients for each parent and the exogenous term. The parents
        are ordered by their index in the graph using `list(dag.predecessors(idx))`.
    coeff_values_requires_grad : list[list[bool]]]
        Whether each coefficient requires a gradient to update it or not.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coeff_values = []
    coeff_values_requires_grad = []
    n_nodes = dag.number_of_nodes()
    for idx in range(n_nodes):
        coeff_values_i = []
        coeff_values_requires_grad_i = []
        num_parents = len(list(dag.predecessors(idx)))
        for pa_idx in range(num_parents):
            random_val = Uniform(min_val, max_val).sample((1,))
            val = random_val
            param = nn.Parameter(val * torch.ones(1), requires_grad=True).to(device)
            coeff_values_i.append(param)
            coeff_values_requires_grad_i.append(True)

        const = torch.ones(1, requires_grad=False).to(device)  # variance param
        coeff_values_i.append(const)
        coeff_values_requires_grad_i.append(False)
        coeff_values.append(nn.ParameterList(coeff_values_i))
        coeff_values_requires_grad.append(coeff_values_requires_grad_i)
    return coeff_values, coeff_values_requires_grad


def set_initial_noise_parameters(
    dag: nx.DiGraph,
    fix_mechanisms: bool,
    intervention_targets: Tensor,
    environments: Tensor,
    min_val: float,
    max_val: float,
    n_dim_per_node: list[int] = None,
    device: torch.device = None,
) -> tuple[list[ParameterList], list[list[bool]]]:
    """Set initial noise parameters for each node's distribution and pass to device.

    Parameters
    ----------
    dag : nx.DiGraph
        Graph structure of the SCM.
    fix_mechanisms : bool
        Whether to allow fix the mechanisms and intervention targets,
        so it is non-trainable.
    intervention_targets : Tensor
        Intervention targets for each distribution.
    environments : Tensor
        The environment index for each intervention target.
        XXX: included to enable selection diagram inputs. We need a function
        that is part of the graph that queries whether a variable/node is "shifted"
        or not given two domain indices.
    min_val : float
        The minimum value for the noise parameter.
    max_val : float
        The maximum value for the noise parameter.
    n_dim_per_node : list[int]
        The number of dimensions for each node in the graph. Default is None,
        corresponding to a single dimension for each node. If a node has more
        than one dimension, it will have a separate noise parameter for each
        dimension and sample from a multivariate distribution.
    device : torch.device
        The device to put the parameters on. Can be 'cpu' or 'cuda'.

    Returns
    -------
    noise_params : list[ParameterList]
        The noise means/stds for each variable in each distribution.
    noise_params_requires_grad : list[list[bool]]]
        Whether each noise mean/std requires a gradient to update it or not.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_dim_per_node is None:
        n_dim_per_node = [1] * dag.number_of_nodes()

    noise_params = []
    noise_params_requires_grad = []
    num_distributions = intervention_targets.shape[0]

    for idx in range(num_distributions):
        noise_means_e = []
        noise_means_requires_grad_e = []
        for node_idx in range(dag.number_of_nodes()):
            n_dims = n_dim_per_node[node_idx]
            is_shifted = intervention_targets[idx][node_idx] == 1
            is_root = len(list(dag.predecessors(node_idx))) == 0

            # fix mechanism if it is not shifted by an intervention
            if fix_mechanisms:
                is_fixed = is_shifted
            else:
                # fix mechanism if it is a root or if it is not shifted
                is_fixed = (is_shifted and not is_root) or (not is_shifted and is_root)
            is_fixed = is_fixed

            # initialize the means
            random_val = Uniform(min_val, max_val).sample((n_dims,))
            val = random_val
            params = (nn.Parameter(val * torch.ones(1), requires_grad=not is_fixed)).to(device)
            noise_means_e.append(params)
            noise_means_requires_grad_e.append(not is_fixed)
        noise_params.append(nn.ParameterList(noise_means_e))
        noise_params_requires_grad.append(noise_means_requires_grad_e)
    return noise_params, noise_params_requires_grad
