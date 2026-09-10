import pytest
import torch

from audio_effect_replicator.device import resolve_device


def test_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_returns_some_device() -> None:
    assert resolve_device(None).type in {"cuda", "mps", "cpu"}
    assert resolve_device("auto") == resolve_device(None)


def test_unavailable_backend_raises() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available on this machine")
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")
