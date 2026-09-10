import os
from pathlib import Path

import numpy as np
import pytest
import torch

from audio_effect_replicator.audio import load_wave
from audio_effect_replicator.legacy import load_keras_checkpoint
from audio_effect_replicator.model import FxReplicator
from audio_effect_replicator.predict import predict

h5py = pytest.importorskip("h5py")


def write_keras_h5(path: Path, model: FxReplicator) -> None:
    """Write FxReplicator weights in the Keras 2.1 CuDNNLSTM h5 layout (inverse of the import)."""
    layers = (("lstm_1", model.lstm1), ("lstm_2", model.lstm2), ("lstm_out", model.lstm_out))
    with h5py.File(path, "w") as f:
        for name, lstm in layers:
            u = lstm.hidden_size
            w_ih, w_hh, b_ih, b_hh = (p.detach().numpy() for p in lstm.all_weights[0])
            kernel = np.concatenate(
                [w_ih[g * u : (g + 1) * u].reshape(-1, u) for g in range(4)], axis=1
            )
            recurrent = np.concatenate([w_hh[g * u : (g + 1) * u] for g in range(4)], axis=1)
            g = f.create_group(f"model_weights/{name}/{name}")
            g["kernel:0"] = kernel
            g["recurrent_kernel:0"] = recurrent
            g["bias:0"] = np.concatenate([b_ih, b_hh])


def test_roundtrip_through_keras_layout(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original = FxReplicator(hidden=8)
    path = tmp_path / "model_000012.h5"
    write_keras_h5(path, original)
    loaded, epoch = load_keras_checkpoint(path)
    assert epoch == 12
    assert not loaded.training
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_epoch_defaults_to_zero_for_other_names(tmp_path: Path) -> None:
    path = tmp_path / "final.h5"
    write_keras_h5(path, FxReplicator(hidden=4))
    assert load_keras_checkpoint(path)[1] == 0


@pytest.mark.skipif(
    not (os.environ.get("AER_2018_H5") and os.environ.get("AER_2018_VAL_DIR")),
    reason="set AER_2018_H5 and AER_2018_VAL_DIR to run against the original 2018 files",
)
def test_reproduces_2018_prediction() -> None:
    val_dir = Path(os.environ["AER_2018_VAL_DIR"])
    model, _ = load_keras_checkpoint(os.environ["AER_2018_H5"])
    x = load_wave(val_dir / "val_x.wav")
    expected = load_wave(val_dir / "predicted.wav")
    pred = predict(model, x, 5280, 480, 16, torch.device("cpu"))
    assert np.abs(pred - expected).max() <= 2e-4
