"""Training loop reproducing the 2018 procedure: random windows, tail MSE, Adam,
best-only checkpoints, early stopping, TensorBoard."""

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from audio_effect_replicator.config import Config
from audio_effect_replicator.data import WindowSampler, load_pairs
from audio_effect_replicator.loss import tail_mse
from audio_effect_replicator.model import FxReplicator, save_checkpoint

log = logging.getLogger(__name__)

LEARNING_RATE = 1e-3


def train(
    config: Config, device: torch.device, out_dir: Path = Path("."), seed: int | None = None
) -> Path:
    """Train until early stopping or `max_epochs`; return the checkpoint directory."""
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)

    train_sampler = WindowSampler(
        load_pairs(config.train_data, config.sample_rate),
        config.input_timesteps,
        config.batch_size,
        rng,
    )
    val_sampler = WindowSampler(
        load_pairs(config.val_data, config.sample_rate),
        config.input_timesteps,
        config.batch_size,
        rng,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = out_dir / "checkpoint" / stamp
    ckpt_dir.mkdir(parents=True)
    writer = SummaryWriter(str(out_dir / "tensorboard" / stamp))

    model = FxReplicator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = _run_steps(
            model, train_sampler, config.steps_per_epoch, config.output_timesteps, device, optimizer
        )
        model.eval()
        with torch.no_grad():
            val_loss = _run_steps(
                model, val_sampler, config.validation_steps, config.output_timesteps, device, None
            )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss < best:
            best = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                ckpt_dir / f"model_{epoch:06d}.pt",
                model,
                config.model,
                config.input_timesteps,
                config.output_timesteps,
                epoch,
                val_loss,
                sample_rate=config.sample_rate,
            )
        else:
            epochs_without_improvement += 1
        log.info(
            "epoch %d train_loss %.6f val_loss %.6f best %.6f", epoch, train_loss, val_loss, best
        )
        if epochs_without_improvement >= config.patience:
            log.info("early stopping after %d epochs", epoch)
            break

    writer.close()
    return ckpt_dir


def _run_steps(
    model: FxReplicator,
    sampler: WindowSampler,
    steps: int,
    output_timesteps: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total = 0.0
    for _ in range(steps):
        x, y = sampler.sample()
        x, y = x.to(device), y.to(device)
        loss = tail_mse(model(x), y, output_timesteps)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / steps
