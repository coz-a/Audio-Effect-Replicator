"""Command line interface: `aer train` and `aer predict`."""

import argparse
import logging
import sys
import wave
from pathlib import Path

from audio_effect_replicator import __version__
from audio_effect_replicator.audio import load_wave, save_wave
from audio_effect_replicator.config import load_config
from audio_effect_replicator.device import resolve_device
from audio_effect_replicator.model import load_checkpoint
from audio_effect_replicator.predict import predict
from audio_effect_replicator.train import train


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aer", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model from paired WAV files")
    p_train.add_argument("-c", "--config", required=True, type=Path, help="config YAML")
    p_train.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p_train.add_argument(
        "--out-dir", default=Path("."), type=Path, help="where checkpoint/ and tensorboard/ go"
    )
    p_train.add_argument("--seed", type=int, default=None)
    p_train.set_defaults(func=_train)

    p_predict = sub.add_parser("predict", help="apply a trained model to a WAV file")
    p_predict.add_argument(
        "-c", "--config", required=True, type=Path, help="config YAML (batch_size)"
    )
    p_predict.add_argument(
        "-i", "--input", required=True, type=Path, help="input WAV (48 kHz mono 16-bit)"
    )
    p_predict.add_argument("-o", "--output", default=Path("predicted.wav"), type=Path)
    p_predict.add_argument("-m", "--model", required=True, type=Path, help="checkpoint (*.pt)")
    p_predict.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p_predict.set_defaults(func=_predict)
    return parser


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = resolve_device(args.device)
    ckpt_dir = train(config, device, out_dir=args.out_dir, seed=args.seed)
    print(f"checkpoints written to {ckpt_dir}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = resolve_device(args.device)
    model, meta = load_checkpoint(args.model, device)
    samples = load_wave(args.input)
    output = predict(
        model, samples, meta["input_timesteps"], meta["output_timesteps"], config.batch_size, device
    )
    save_wave(output, args.output)
    print(f"wrote {args.output} ({len(output)} samples)")
    return 0
