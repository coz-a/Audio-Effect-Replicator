# Audio Effect Replicator

LSTM-based black-box modeling of audio effects.

Originally developed and publicly released in February 2018.

The original February 2018 implementation is preserved at the
[`2018-original`](https://github.com/coz-a/Audio-Effect-Replicator/tree/2018-original) tag.

## Historical Background

This project was originally developed in February 2018 as an experiment in
modeling guitar amplifier / audio-effect behavior directly from paired input
and output waveforms using an LSTM neural network.

The idea and the results are described (in Japanese) in the accompanying
article:

- [Machine Learning-Based Guitar Amp Modeling](https://qiita.com/coz-a/items/aeab3c52e3f12ba52a8b) (Qiita, 2018-02-04)

## Architecture

The model is three stacked LSTM layers that map a dry input waveform to the
effected output waveform, producing one output sample per input sample. It is
implemented with PyTorch `nn.LSTM` (cuDNN on CUDA); the 2018 version used Keras
`CuDNNLSTM` with the same layer sizes.

```
Input waveform (48 kHz, mono)
        |
LSTM (64 units)
        |
LSTM (64 units)
        |
LSTM (1 unit)
        |
Output waveform (48 kHz, mono)
```

- **Data:** pairs of 48 kHz / mono / 16-bit PCM WAV files (dry input and
  effected output), listed in the config file.
- **Training:** random windows of `input_timesteps` (5280) samples are drawn
  from the training pairs. The loss is the mean squared error over the last
  `output_timesteps` (480) samples of each window only; the preceding 4800
  samples act as a warm-up for the LSTM state. Adam, learning rate 1e-3.
- **Inference:** the input is zero-padded with 4800 samples in front and
  processed with a sliding window (window 5280, hop 480). For each window the
  last 480 output samples are kept and concatenated; the output has exactly
  the length of the input.

Weights are initialized as in Keras 2.1 (glorot-uniform input kernels,
orthogonal recurrent kernels, forget-gate bias 1); with PyTorch's default
initialization this stack barely trains. Runs are still not bit-identical to
the 2018 Keras runs.

## Installation

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
PyTorch is installed through one of two extras:

```sh
uv sync --extra cu130   # NVIDIA GPU (CUDA 13)
uv sync --extra cpu     # CPU only
```

## Training

Copy `configs/default.yml`, point `train_data` / `val_data` at your WAV pairs,
then:

```sh
uv run aer train -c configs/default.yml
```

Checkpoints are written to `checkpoint/<timestamp>/model_<epoch>.pt` (epoch
zero-padded to six digits, best validation loss only) and TensorBoard logs to
`tensorboard/<timestamp>/`. Training stops early after `patience` epochs
without improvement in validation loss. `--device cpu|cuda|mps` overrides the
automatic choice; `--seed N` makes the run reproducible.

## Inference

```sh
uv run aer predict -c configs/default.yml -i input.wav -o predicted.wav -m checkpoint/<timestamp>/model_000031.pt
```

`input.wav` must be 48 kHz / mono / 16-bit PCM (other formats are rejected).
The result is written as 48 kHz / mono / 16-bit PCM with the same number of
samples. No pre-trained weights are included in this repository.

## Development

```sh
uv sync --extra cu130
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest
```

## Legacy Version

The original February 2018 scripts (`train.py`, `predict.py`,
`fx_replicator.py`) and their `requirements.txt` (TensorFlow GPU 1.5,
Keras 2.1) are kept at the `2018-original` tag:

```sh
git checkout 2018-original
```

> The original dependencies are retained for historical reference only.
> They should not be used in production or security-sensitive environments.

Legacy 2018 environments are not supported.

## License

MIT License. See [LICENSE](LICENSE).
