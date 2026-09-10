"""Three stacked LSTM layers (64, 64, 1), the 2018 architecture in PyTorch."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1


class FxReplicator(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_out = nn.LSTM(hidden, 1, batch_first=True)
        for lstm in (self.lstm1, self.lstm2, self.lstm_out):
            _init_like_keras(lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        y, _ = self.lstm_out(h)
        return y


def _init_like_keras(lstm: nn.LSTM) -> None:
    """Keras 2.1 LSTM defaults: glorot_uniform input kernel, orthogonal recurrent kernel
    (per gate), zero biases with the forget gate bias set to 1. With PyTorch's default
    uniform(-1/sqrt(hidden), 1/sqrt(hidden)) this 3-layer stack barely trains."""
    h = lstm.hidden_size
    weight_ih, weight_hh, bias_ih, bias_hh = lstm.all_weights[0]
    with torch.no_grad():
        nn.init.xavier_uniform_(weight_ih)
        for gate in range(4):
            nn.init.orthogonal_(weight_hh[gate * h : (gate + 1) * h])
        bias_ih.zero_()
        bias_hh.zero_()
        bias_ih[h : 2 * h] = 1.0  # torch gate order: input, forget, cell, output


def save_checkpoint(
    path: str | Path,
    model: FxReplicator,
    input_timesteps: int,
    output_timesteps: int,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "hidden": model.hidden,
            "input_timesteps": input_timesteps,
            "output_timesteps": output_timesteps,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[FxReplicator, dict[str, Any]]:
    meta: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    model = FxReplicator(hidden=meta["hidden"]).to(device)
    model.load_state_dict(meta["model_state_dict"])
    model.eval()
    return model, meta
