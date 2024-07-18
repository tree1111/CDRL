import pytest
import torch

from gendis.noncausal.flows import Reshape


def test_forward_inverse_consistency():
    in_shape = (2, 4, 4)
    out_shape = (4, 2, 4)
    reshape_instance = Reshape(in_shape, out_shape)
    batch_size = 2
    input_tensor = torch.randn(batch_size, *reshape_instance.in_shape)

    reshaped_tensor, _ = reshape_instance.forward(input_tensor)
    restored_tensor, _ = reshape_instance.inverse(reshaped_tensor)

    assert torch.allclose(input_tensor, restored_tensor)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        Reshape((2, 4, 4), (3, 3, 3))
