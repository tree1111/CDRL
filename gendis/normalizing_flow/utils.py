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


def sample_invertible_matrix(size, min_val, max_val, max_samples=1000):
    """Samples a square invertible matrix."""
    idx = 0
    while True and idx < max_samples:
        # Sample a random matrix with uniform entries
        matrix = (max_val - min_val) * torch.rand((size, size)) + min_val
        idx += 1

        # Check if the matrix is invertible by computing its determinant
        if torch.det(matrix) != 0:
            return matrix
    raise ValueError(f"Could not find an invertible matrix after {max_samples} samples.")


def set_initial_edge_coeffs(
    dag: nx.DiGraph,
    min_val: float = -1,
    max_val: float = 1,
    cluster_mapping=None,
    use_matrix: bool = False,
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
    cluster_mapping : dict
        Mapping of nodes to latent indices. Default is None,
        which assumes each node has a single latent index, and
        each edge is therefore a single coefficient. If the
        latent indices are multivariate, then an invertible matrix
        is sampled for each "edge", which maps the incoming latent
        cluster node to the outgoing node.
        This assumes that each cluster mapping is a list of indices
        that correspond to the latent dimensions.
    use_matrix : bool
        Whether to use a matrix to represent the edge coefficients.
        Default is False, which corresponds to a vector of coefficients
        if cluster_mapping is not None. This then applies a element-by-element
        multiplication of the latent cluster node with the edge coefficient.
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
            if cluster_mapping is None:
                random_val = Uniform(min_val, max_val).sample((1,))
                val = random_val
                param = nn.Parameter(val * torch.ones(1), requires_grad=True).to(device)
            elif cluster_mapping is not None and use_matrix:
                if len(cluster_mapping[idx]) != len(cluster_mapping[pa_idx]):
                    raise ValueError(
                        "The number of dimensions for the latent clusters must be the same if "
                        "using a matrix to represent the edge coefficients."
                        f"{idx}: {len(cluster_mapping[idx])} != {pa_idx}: {len(cluster_mapping[pa_idx])}."
                    )
                # sample a random invertible matrix
                matrix = sample_invertible_matrix(len(cluster_mapping[idx]), min_val, max_val)
                param = nn.Parameter(matrix, requires_grad=True).to(device)
            else:
                # sample a random vector with each value between min_val and max_val
                vector = (max_val - min_val) * torch.rand((len(cluster_mapping[idx]),)) + min_val
                param = nn.Parameter(vector, requires_grad=True).to(device)

            coeff_values_i.append(param)
            coeff_values_requires_grad_i.append(True)

        if cluster_mapping is None:
            const = torch.ones(1, requires_grad=False).to(device)  # variance param
        else:
            const = nn.Parameter(torch.ones(len(cluster_mapping[idx])), requires_grad=False).to(
                device
            )

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
    cluster_mapping=None,
    use_matrix: bool = False,
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

        # this goes in the order of the nodes in the graph
        # stored as an adjacency matrix
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
            params = (nn.Parameter(random_val * torch.ones(1), requires_grad=not is_fixed)).to(
                device
            )
            noise_means_e.append(params)
            noise_means_requires_grad_e.append(not is_fixed)
        noise_params.append(nn.ParameterList(noise_means_e))
        noise_params_requires_grad.append(noise_means_requires_grad_e)
    return noise_params, noise_params_requires_grad
