import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gendis.model import LinearNeuralClusteredASCMFlow, NonlinearNeuralClusteredASCMFlow


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


@pytest.mark.parametrize(
    "MODEL_CLS", [NonlinearNeuralClusteredASCMFlow, LinearNeuralClusteredASCMFlow]
)
def test_model_initialization(MODEL_CLS, sample_graph, sample_cluster_sizes):
    model = MODEL_CLS(
        graph=sample_graph,
        cluster_sizes=sample_cluster_sizes,
        intervention_targets_per_distr=torch.rand(size=(2, 2)),
    )
    assert model.graph.shape == (2, 2)
    assert model.latent_dim == 2


@pytest.mark.parametrize(
    "MODEL_CLS", [NonlinearNeuralClusteredASCMFlow, LinearNeuralClusteredASCMFlow]
)
def test_forward_pass(MODEL_CLS, sample_graph, sample_cluster_sizes):
    model = MODEL_CLS(
        graph=sample_graph,
        cluster_sizes=sample_cluster_sizes,
        intervention_targets_per_distr=torch.rand(size=(2, 2)),
    )
    x = torch.rand(5, 2)
    v_hat = model(x)
    assert v_hat.shape == x.shape


@pytest.mark.parametrize(
    "MODEL_CLS", [NonlinearNeuralClusteredASCMFlow, LinearNeuralClusteredASCMFlow]
)
def test_training_step(MODEL_CLS, sample_graph, sample_cluster_sizes, sample_data):
    model = MODEL_CLS(
        graph=sample_graph,
        cluster_sizes=sample_cluster_sizes,
        intervention_targets_per_distr=torch.rand(size=(2, 2)),
    )
    batch = next(iter(sample_data))
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize(
    "MODEL_CLS", [NonlinearNeuralClusteredASCMFlow, LinearNeuralClusteredASCMFlow]
)
def test_validation_step(MODEL_CLS, sample_graph, sample_cluster_sizes, sample_data):
    model = MODEL_CLS(
        graph=sample_graph,
        cluster_sizes=sample_cluster_sizes,
        intervention_targets_per_distr=torch.rand(size=(2, 2)),
    )
    batch = next(iter(sample_data))
    output = model.validation_step(batch, 0)
    assert isinstance(output, dict)
    assert "log_prob" in output
    assert "log_prob_gt" in output
    assert "v" in output
    assert "v_hat" in output


# @pytest.mark.parametrize(
#     "MODEL_CLS", [NonlinearNeuralClusteredASCMFlow, LinearNeuralClusteredASCMFlow]
# )
# def test_configure_optimizers(MODEL_CLS, sample_graph, sample_cluster_sizes):
#     if MODEL_CLS == LinearNeuralClusteredASCMFlow:
#         with pytest.raises(RuntimeError):
#             model = MODEL_CLS(
#                 graph=sample_graph,
#                 cluster_sizes=sample_cluster_sizes,
#                 lr_scheduler="cosine",
#             )
#         model = MODEL_CLS(
#             graph=sample_graph,
#             intervention_targets_per_distr=torch.rand(size=(2, 2)),
#             cluster_sizes=sample_cluster_sizes,
#             lr_scheduler="cosine",
#         )
#     else:
#         model = MODEL_CLS(
#             graph=sample_graph,
#             cluster_sizes=sample_cluster_sizes,
#             lr_scheduler="cosine",
#         )
#     optimizers = model.configure_optimizers()
#     assert isinstance(optimizers, dict)
#     assert "optimizer" in optimizers
#     assert "lr_scheduler" in optimizers
