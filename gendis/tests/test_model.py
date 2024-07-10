import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gendis.model import NeuralClusteredASCMFlow


@pytest.fixture
def sample_graph():
    return np.array([[0, 1], [0, 0]])


@pytest.fixture
def sample_cluster_sizes():
    return [1, 1]


@pytest.fixture
def sample_data():
    x = torch.rand(10, 2)
    v = torch.rand(10, 2)
    u = torch.rand(10, 2)
    e = torch.zeros(10, 1)
    int_target = torch.zeros(10, 2)
    log_prob_gt = torch.zeros(10)
    dataset = TensorDataset(x, v, u, e, int_target, log_prob_gt)
    return DataLoader(dataset, batch_size=2)


@pytest.mark.parametrize("MODEL_CLS", [NeuralClusteredASCMFlow])
def test_forward_pass(MODEL_CLS):
    model = MODEL_CLS()
    x = torch.rand(5, 2)
    v_hat = model(x)
    assert v_hat.shape == x.shape


@pytest.mark.parametrize("MODEL_CLS", [NeuralClusteredASCMFlow])
def test_training_step(MODEL_CLS, sample_data):
    model = MODEL_CLS()
    batch = next(iter(sample_data))
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize("MODEL_CLS", [NeuralClusteredASCMFlow])
def test_validation_step(MODEL_CLS, sample_data):
    model = MODEL_CLS()
    batch = next(iter(sample_data))
    output = model.validation_step(batch, 0)
    assert isinstance(output, dict)
    assert "log_prob" in output
    assert "log_prob_gt" in output
    assert "v" in output
    assert "v_hat" in output
