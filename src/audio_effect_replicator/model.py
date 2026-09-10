"""Three stacked LSTM layers (64, 64, 1), the 2018 architecture in PyTorch."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from audio_effect_replicator.config import Config

CHECKPOINT_VERSION = 1


class FxReplicator(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_out = nn.LSTM(hidden, 1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        y, _ = self.lstm_out(h)
        return y


def save_checkpoint(
    path: str | Path, model: FxReplicator, config: Config, epoch: int, val_loss: float
) -> None:
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "hidden": model.hidden,
            "input_timesteps": config.input_timesteps,
            "output_timesteps": config.output_timesteps,
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
