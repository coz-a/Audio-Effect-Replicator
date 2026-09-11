"""Command line interface: train, predict, evaluate, import-keras and fetch-dataset."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from audio_effect_replicator import __version__
from audio_effect_replicator.audio import SAMPLE_RATE, load_wave, save_wave
from audio_effect_replicator.config import load_config
from audio_effect_replicator.datasets import (
    fetch_dataset,
    list_datasets,
    load_manifest,
    packaged_manifest,
)
from audio_effect_replicator.device import resolve_device
from audio_effect_replicator.evaluate import evaluate_prediction, time_inference
from audio_effect_replicator.legacy import load_keras_checkpoint
from audio_effect_replicator.metrics import parameter_count
from audio_effect_replicator.model import load_checkpoint, save_checkpoint
from audio_effect_replicator.predict import predict
from audio_effect_replicator.train import train


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
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

    p_eval = sub.add_parser(
        "evaluate", help="score a model or an existing prediction against a target WAV"
    )
    p_eval.add_argument("-t", "--target", required=True, type=Path, help="reference output WAV")
    source = p_eval.add_mutually_exclusive_group(required=True)
    source.add_argument("-m", "--model", type=Path, help="checkpoint (*.pt) to run on --input")
    source.add_argument("--prediction", type=Path, help="already predicted WAV to score")
    p_eval.add_argument("-i", "--input", type=Path, help="input WAV for --model")
    p_eval.add_argument(
        "-c", "--config", type=Path, help="config YAML (batch_size only, default 16)"
    )
    p_eval.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p_eval.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="expected rate for --prediction (a model uses its checkpoint's rate)",
    )
    p_eval.add_argument("--json", type=Path, help="also write the scores to this JSON file")
    p_eval.set_defaults(func=_evaluate)

    p_import = sub.add_parser("import-keras", help="convert a 2018 Keras checkpoint (*.h5) to *.pt")
    p_import.add_argument("h5", type=Path)
    p_import.add_argument("-o", "--output", required=True, type=Path)
    p_import.add_argument("--input-timesteps", type=int, default=5280)
    p_import.add_argument("--output-timesteps", type=int, default=480)
    p_import.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p_import.set_defaults(func=_import_keras)

    p_fetch = sub.add_parser("fetch-dataset", help="download a benchmark dataset into --dest")
    p_fetch.add_argument("name", nargs="?", help="dataset name (see --list)")
    p_fetch.add_argument("--dest", default=Path("data"), type=Path)
    p_fetch.add_argument(
        "--manifest", type=Path, help="use this manifest file instead of a packaged one"
    )
    p_fetch.add_argument("--list", action="store_true", help="list packaged datasets and exit")
    p_fetch.set_defaults(func=_fetch_dataset)
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
    samples = load_wave(args.input, meta["sample_rate"])
    output = predict(
        model, samples, meta["input_timesteps"], meta["output_timesteps"], config.batch_size, device
    )
    save_wave(output, args.output, meta["sample_rate"])
    print(f"wrote {args.output} ({len(output)} samples)")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if args.model is not None:
        if args.input is None:
            raise ValueError("-i/--input is required together with -m/--model")
        device = resolve_device(args.device)
        batch_size = load_config(args.config).batch_size if args.config else 16
        model, meta = load_checkpoint(args.model, device)
        rate = meta["sample_rate"]
        target = load_wave(args.target, rate)
        samples = load_wave(args.input, rate)
        timesteps = (meta["input_timesteps"], meta["output_timesteps"])
        pred = predict(model, samples, *timesteps, batch_size, device)
        result: dict[str, object] = {
            "checkpoint": str(args.model),
            "epoch": meta["epoch"],
            "parameters": parameter_count(model),
            "device": device.type,
            "sample_rate": rate,
            **evaluate_prediction(pred, target),
            **time_inference(model, samples, *timesteps, batch_size, device, sample_rate=rate),
        }
    else:
        rate = args.sample_rate
        target = load_wave(args.target, rate)
        prediction = load_wave(args.prediction, rate)
        result = {
            "prediction": str(args.prediction),
            "sample_rate": rate,
            **evaluate_prediction(prediction, target),
        }
    for key, value in result.items():
        print(f"{key:18s} {value:.6g}" if isinstance(value, float) else f"{key:18s} {value}")
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


def _import_keras(args: argparse.Namespace) -> int:
    model, epoch = load_keras_checkpoint(args.h5)
    save_checkpoint(
        args.output,
        model,
        args.input_timesteps,
        args.output_timesteps,
        epoch,
        math.nan,
        sample_rate=args.sample_rate,
    )
    print(f"wrote {args.output} (epoch {epoch}, {parameter_count(model)} parameters)")
    return 0


def _fetch_dataset(args: argparse.Namespace) -> int:
    if args.list:
        print("\n".join(list_datasets()))
        return 0
    if args.manifest is None and args.name is None:
        raise ValueError("give a dataset name or --manifest (see --list)")
    manifest = load_manifest(args.manifest or packaged_manifest(args.name))
    root = fetch_dataset(manifest, args.dest)
    print(f"{manifest.name}: {len(manifest.files)} files in {root}")
    print(f"license: {manifest.license}")
    print(f"citation: {manifest.citation}")
    return 0
