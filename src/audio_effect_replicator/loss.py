"""Mean squared error over the last `timesteps` samples only (the 2018 LossFunc)."""

import torch
from torch.nn.functional import mse_loss


def tail_mse(y_pred: torch.Tensor, y_true: torch.Tensor, timesteps: int) -> torch.Tensor:
    return mse_loss(y_pred[:, -timesteps:], y_true[:, -timesteps:])
