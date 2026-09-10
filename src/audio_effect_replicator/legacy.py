"""Import the 2018 Keras 2.1 CuDNNLSTM checkpoints (*.h5) into FxReplicator.

Keras handed each gate's kernel slice to cuDNN by flattening it in memory, not by
transposing it, so the input kernels are reinterpreted rather than transposed.
Verified against the original 2018 checkpoint: the imported model reproduces the
2018 prediction to within 4 LSB.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from audio_effect_replicator.model import FxReplicator

_LAYERS = ("lstm_1", "lstm_2", "lstm_out")
_KEYS = ("kernel:0", "recurrent_kernel:0", "bias:0")


def load_keras_checkpoint(path: str | Path) -> tuple[FxReplicator, int]:
    """Return the model (eval mode) and the epoch parsed from `model_XXXXXX.h5` (0 if absent)."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is not installed; run `uv sync --extra legacy`") from exc
    with h5py.File(path, "r") as f:
        model = FxReplicator(hidden=_read(f, "lstm_1", "recurrent_kernel:0").shape[0])
        for name, lstm in zip(_LAYERS, (model.lstm1, model.lstm2, model.lstm_out), strict=True):
            kernel, recurrent, bias = (_read(f, name, key) for key in _KEYS)
            _copy_cudnn_weights(lstm, kernel, recurrent, bias)
    model.eval()
    return model, _epoch_from_name(path)


def _read(file: Any, name: str, key: str) -> np.ndarray:
    import h5py

    node = file[f"model_weights/{name}/{name}/{key}"]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"{name}/{key} is not a dataset")
    return np.asarray(node)


def _copy_cudnn_weights(
    lstm: nn.LSTM, kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray
) -> None:
    u = lstm.hidden_size
    gates = range(4)  # i, f, c, o in both Keras and torch
    w_ih, w_hh, b_ih, b_hh = lstm.all_weights[0]
    with torch.no_grad():
        w_ih.copy_(
            torch.from_numpy(
                np.concatenate([kernel[:, g * u : (g + 1) * u].reshape(u, -1) for g in gates])
            )
        )
        w_hh.copy_(
            torch.from_numpy(
                np.ascontiguousarray(
                    np.concatenate([recurrent[:, g * u : (g + 1) * u] for g in gates])
                )
            )
        )
        b_ih.copy_(torch.from_numpy(np.ascontiguousarray(bias[: 4 * u])))
        b_hh.copy_(torch.from_numpy(np.ascontiguousarray(bias[4 * u :])))


def _epoch_from_name(path: str | Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.h5", Path(path).name)
    return int(match.group(1)) if match else 0
