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

- **Data:** pairs of mono WAV files (dry input and effected output) at the
  configured sample rate, 48 kHz in the 2018 setup, listed in the config file.
- **Training:** random windows of `input_timesteps` (5280) samples are drawn
  from the training pairs. The loss is the mean squared error over the last
  `output_timesteps` (480) samples of each window only; the preceding 4800
  samples act as a warm-up for the LSTM state. Adam, learning rate 1e-3.
- **Inference:** the input is zero-padded with 4800 samples in front and
  processed with a sliding window (window 5280, hop 480). For each window the
  last 480 output samples are kept and concatenated; the output has exactly
  the length of the input.

Two other architectures can be selected in the config: `wright` is the single
LSTM layer with a linear output of Wright et al. (2019), and `skip` is that
model predicting only the difference from its input. `lstm2018` is the default
and reproduces the original stack.

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

Add `--extra legacy` to install `h5py`, needed only for `aer import-keras`.

## Datasets

Any pair of mono WAV files (16/24/32-bit PCM or 32-bit float) can be used;
set `sample_rate` in the config if it is not 48 kHz. Public benchmark data can
be fetched with a checksum-verified download:

```sh
uv run aer fetch-dataset --list
uv run aer fetch-dataset wright2019      # -> data/wright2019/
```

If [axel](https://github.com/axel-download-accelerator/axel) is on the PATH it
is used automatically, which is far faster on hosts that throttle a single
connection, and it resumes an interrupted download.

`wright2019` is the Blackstar HT-1 / Big Muff Pi data from Wright, Damskägg
and Välimäki, "Real-Time Black-Box Modelling with Recurrent Neural Networks"
(DAFx 2019), 44.1 kHz, with the train / val / test split of the original
repository. It is licensed CC BY-NC 4.0: fine for research and comparisons,
not for commercial models. `configs/wright2019-ht1.yml` and
`configs/wright2019-muff.yml` train on it with the 2018 settings.

`tonetwist-afx-analog`, `tonetwist-afx-digital`, `tonetwist-afx-analog-parametric`
and `tonetwist-afx-digital-parametric` are the subsets of the ToneTwisT AFx
dataset (Comunità, Steinmetz and Reiss, 2025): about 50 hardware and plugin
effects as dry/wet pairs at 48 kHz / 32-bit float, 72 GB in total, all
CC BY-NC 4.0. Each Zenodo archive is extracted into `data/tonetwist-afx/<name>/`;
split archives need 7-Zip (`7z`) on the PATH.

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

The model, the loss and gradient clipping are set in the config, or overridden
on the command line for a sweep:

```yaml
model:
  type: skip        # lstm2018 (default), wright or skip
  hidden: 64
loss:
  type: esr         # tail_mse (default), esr or esr_preemphasis
learning_rate: 0.001
grad_clip: 0        # 0 disables clipping, as in the 2018 method
```

```sh
uv run aer train -c configs/wright2019-ht1.yml --model wright --grad-clip 20
```

`esr` is the error-to-signal ratio and `esr_preemphasis` adds the first-order
high-pass and the DC term of the DAFx 2019 loss. All three are computed over
the same last `output_timesteps` samples of each window.

Backpropagating through a 5280-sample window occasionally produces an
exploding gradient that drives the LSTM to a silent output it never recovers
from: a gradient norm of 1094 was measured against a typical maximum of 13,
and early stopping then keeps the checkpoint from before the collapse.
`grad_clip` bounds that step. It is off by default because the 2018 method did
not clip; the benchmarks below use 20, above the normal range and far below
the spike.

## Inference

```sh
uv run aer predict -c configs/default.yml -i input.wav -o predicted.wav -m checkpoint/<timestamp>/model_<epoch>.pt
```

`input.wav` must be mono at the checkpoint's sample rate (other files are
rejected); 16/24/32-bit PCM and 32-bit float are accepted. The result is
written as 16-bit PCM at the same rate with the same number of samples. No
pre-trained weights are included in this repository.

## Evaluation

Score a checkpoint on a held-out pair, or score an existing prediction:

```sh
uv run aer evaluate -t val_y.wav -m checkpoint/<timestamp>/model_<epoch>.pt -i val_x.wav
uv run aer evaluate -t val_y.wav --prediction predicted.wav
```

Reported values (add `--json scores.json` to save them):

- `mse`: mean squared error against the target
- `esr`: error-to-signal ratio, sum of squared error over the target's energy
- `lsd_db`: log-spectral distance, RMS difference of the log-magnitude STFTs
  (2048 / 512, both signals rounded to 16-bit PCM first) in dB
- `parameters`, `inference_seconds`, `realtime_factor` (audio seconds per
  wall-clock second) when a model is given

## Benchmarks

Test-set scores on the `wright2019` data: median of three seeds, range in
brackets. Training follows the 2018 procedure (random windows, state reset per
window, loss over the last 480 samples) with `--grad-clip 20`. No pre-trained
weights are shipped.

| Device | Model | Parameters | ESR | LSD (dB) |
| --- | --- | --- | --- | --- |
| Blackstar HT-1 | LSTM 64-64-1 (2018 method) | 50,700 | 3.1 % (2.5-3.7) | 14.0 |
| Blackstar HT-1 | Wright LSTM-64 | 17,217 | 2.7 % (2.5-4.2) | 10.3 |
| Blackstar HT-1 | Skip LSTM-64 | 17,217 | 3.3 % (3.2-4.7) | 10.5 |
| Blackstar HT-1 | Skip LSTM-64, ESR + pre-emphasis loss | 17,217 | 6.5 % (5.0-17.5) | 11.3 |
| Big Muff Pi | LSTM 64-64-1 (2018 method) | 50,700 | 10.0 % (9.4-11.1) | 8.6 |
| Big Muff Pi | Wright LSTM-64 | 17,217 | 25.2 % (12.6-27.5) | 19.4 |
| Big Muff Pi | Skip LSTM-64 | 17,217 | 17.4 % (11.8-29.0) | 12.9 |
| Big Muff Pi | Skip LSTM-64, ESR + pre-emphasis loss | 17,217 | 15.4 % (11.0-24.1) | 12.5 |

On the amplifier a single LSTM layer matches the three-layer stack with a
third of the parameters and a clearly lower spectral error. On the fuzz pedal
it does not: the single-layer models land anywhere between 12 % and 29 %
depending on the seed, while the three-layer stack stays within 9-11 %. The
pre-emphasised loss does not help under this training procedure.

For reference, Wright et al. (2019) report test ESR of 1.8 % (HT-1) and 4.1 %
(Big Muff) for an LSTM-64 with a linear output, and 0.79 % / 9.2 % for their
best WaveNet models. They trained on five control settings with a conditioning
input and with a stateful procedure that is not reproduced here, so the
comparison is indicative only.

## Development

```sh
uv sync --extra cpu --extra legacy   # or --extra cu130 on an NVIDIA GPU
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest
```

## Legacy Version

The original February 2018 scripts (`train.py`, `predict.py`,
`fx_replicator.py`) and their `requirements.txt` (TensorFlow GPU 1.5,
Keras 2.1) are kept at the `2018-original` tag:

```sh
git checkout 2018-original
```

Checkpoints saved by the 2018 code (`*.h5`) can be converted and scored with
the current code (needs the `legacy` extra):

```sh
uv run aer import-keras model_000031.h5 -o model_000031.pt
```

> The original dependencies are retained for historical reference only.
> They should not be used in production or security-sensitive environments.

Legacy 2018 environments are not supported.

## License

MIT License. See [LICENSE](LICENSE).
