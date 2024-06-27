import pytest
import networkx as nx
import torch
from torch.nn import ParameterList

from gendis.normalizing_flow.utils import (
    set_initial_edge_coeffs,
    set_initial_noise_parameters,
)


def test_set_initial_edge_coeffs_basic():
    dag = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    min_val = -1.0
    max_val = 1.0
    device = torch.device("cpu")

    coeff_values, coeff_values_requires_grad = set_initial_edge_coeffs(
        dag, min_val, max_val, device
    )

    assert len(coeff_values) == 3  # Three nodes in the graph
    for i in range(3):
        assert isinstance(coeff_values[i], ParameterList)
        assert len(coeff_values[i]) == (len(list(dag.predecessors(i))) + 1)
        for j in range(len(coeff_values[i])):
            assert coeff_values[i][j].device == device
            if j < len(list(dag.predecessors(i))):
                assert coeff_values_requires_grad[i][j] is True
                assert min_val <= coeff_values[i][j].item() <= max_val
            else:
                assert coeff_values_requires_grad[i][j] is False


def test_set_initial_edge_coeffs_device():
    dag = nx.DiGraph([(0, 1)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coeff_values, _ = set_initial_edge_coeffs(dag, device=device)
    for coeff_list in coeff_values:
        for coeff in coeff_list:
            assert coeff.device == device


@pytest.mark.parametrize("min_val, max_val", [(-2, 2), (-0.5, 0.5)])
def test_set_initial_edge_coeffs_value_range(min_val, max_val):
    dag = nx.DiGraph([(0, 1), (1, 2)])
    coeff_values, _ = set_initial_edge_coeffs(dag, min_val, max_val, torch.device("cpu"))

    for coeff_list in coeff_values:
        for coeff in coeff_list[:-1]:  # Exclude the last constant term
            assert min_val <= coeff.item() <= max_val


def test_set_initial_noise_parameters_basic():
    dag = nx.DiGraph([(0, 1), (1, 2)])
    fix_mechanisms = True
    intervention_targets = torch.tensor([[1, 0, 1]])
    environments = torch.tensor([0])
    min_val = -1.0
    max_val = 1.0
    device = torch.device("cpu")

    noise_params, noise_params_requires_grad = set_initial_noise_parameters(
        dag,
        fix_mechanisms,
        intervention_targets,
        environments,
        min_val,
        max_val,
        device=device,
    )

    assert len(noise_params) == 1  # One distribution
    assert len(noise_params[0]) == 3  # Three nodes in the graph
    for param in noise_params[0]:
        assert param.device == device


def test_set_initial_noise_parameters_device():
    dag = nx.DiGraph([(0, 1)])
    fix_mechanisms = False
    intervention_targets = torch.tensor([[0, 1]])
    environments = torch.tensor([0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_params, _ = set_initial_noise_parameters(
        dag, fix_mechanisms, intervention_targets, environments, -1, 1, device=device
    )
    for param_list in noise_params:
        for param in param_list:
            assert param.device == device


@pytest.mark.parametrize("min_val, max_val", [(-2, 2), (-0.5, 0.5)])
def test_set_initial_noise_parameters_value_range(min_val, max_val):
    dag = nx.DiGraph([(0, 1), (1, 2)])
    fix_mechanisms = False
    intervention_targets = torch.tensor([[0, 1, 0]])
    environments = torch.tensor([0])
    device = torch.device("cpu")

    noise_params, _ = set_initial_noise_parameters(
        dag,
        fix_mechanisms,
        intervention_targets,
        environments,
        min_val,
        max_val,
        device=device,
    )

    for param_list in noise_params:
        for param in param_list:
            assert min_val <= param.item() <= max_val


def test_set_initial_noise_parameters_fix_mechanisms():
    dag = nx.DiGraph([(0, 1)])
    fix_mechanisms = True
    intervention_targets = torch.tensor([[1, 1]])
    environments = torch.tensor([0])
    device = torch.device("cpu")

    noise_params, noise_params_requires_grad = set_initial_noise_parameters(
        dag, fix_mechanisms, intervention_targets, environments, -1, 1, device=device
    )

    for param_list, requires_grad_list in zip(noise_params, noise_params_requires_grad):
        for param, requires_grad in zip(param_list, requires_grad_list):
            assert param.requires_grad == requires_grad


def test_set_initial_noise_parameters_n_dim_per_node():
    dag = nx.DiGraph([(0, 1), (1, 2)])
    fix_mechanisms = False
    intervention_targets = torch.tensor([[1, 0, 1]])
    environments = torch.tensor([0])
    min_val = -1.0
    max_val = 1.0
    n_dim_per_node = [3, 1, 2]  # Different dimensions for nodes 0, 1, and 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_params, noise_params_requires_grad = set_initial_noise_parameters(
        dag,
        fix_mechanisms,
        intervention_targets,
        environments,
        min_val,
        max_val,
        n_dim_per_node,
        device,
    )

    for i, param_list in enumerate(noise_params):
        print(f"Node {i} noise parameters:")
        for param in param_list:
            print(param)
