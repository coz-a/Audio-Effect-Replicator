import numpy as np
import pytest
import torch

from audio_effect_replicator.model import FxReplicator
from audio_effect_replicator.predict import predict

CPU = torch.device("cpu")


@pytest.mark.parametrize("length", [0, 1, 100, 479, 480, 481, 7680, 7681, 48000])
def test_output_length_equals_input_length(length: int) -> None:
    model = FxReplicator(hidden=4)
    samples = np.zeros(length, np.float32)
    out = predict(model, samples, 5280, 480, 16, CPU)
    assert out.dtype == np.float32
    assert out.shape == (length,)


def test_batch_size_does_not_change_result() -> None:
    torch.manual_seed(0)
    model = FxReplicator(hidden=4)
    samples = np.random.default_rng(0).standard_normal(3000).astype(np.float32)
    a = predict(model, samples, 64, 16, 1, CPU)
    b = predict(model, samples, 64, 16, 32, CPU)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_windows_cover_input_in_order() -> None:
    """With an identity model, each output block comes from the matching input block."""

    class Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    samples = np.arange(1, 1001, dtype=np.float32)
    out = predict(Identity(), samples, 64, 16, 8, CPU)
    np.testing.assert_array_equal(out, samples)
