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

The model is three stacked LSTM layers (Keras `CuDNNLSTM`) that map a dry
input waveform to the effected output waveform, producing one output sample
per input sample.

```
Input waveform (48 kHz, mono)
        |
LSTM (64 units, return_sequences)
        |
LSTM (64 units, return_sequences)
        |
LSTM (1 unit, return_sequences)
        |
Output waveform (48 kHz, mono)
```

- **Data:** pairs of 48 kHz / mono / 16-bit PCM WAV files (dry input and
  effected output), listed in `config.yml`.
- **Training:** random windows of `input_timesteps` (5280) samples are drawn
  from the training pairs. The loss is the mean squared error over the last
  `output_timesteps` (480) samples of each window only; the preceding 4800
  samples act as a warm-up for the LSTM state.
- **Inference:** the input is zero-padded with 4800 samples in front and
  processed with a sliding window (window 5280, hop 480). For each window the
  last 480 output samples are kept and concatenated to form the output
  waveform.

## Installation

The original package dependencies are pinned in `requirements.txt`
(`tensorflow-gpu==1.5.0`, `Keras==2.1.3` and others). The model uses Keras
`CuDNNLSTM`, so an NVIDIA GPU with cuDNN is required.

```sh
pip install -r requirements.txt
```

> The original dependencies are retained for historical reference only.
> They should not be used in production or security-sensitive environments.

Legacy 2018 environments are not supported.

## Training

Edit `config.yml` to point at your training / validation WAV pairs, then:

```sh
python train.py -c config.yml
```

Checkpoints are written to `checkpoint/<timestamp>/model_<epoch>.h5` with the
epoch zero-padded to six digits (e.g. `model_000031.h5`), keeping only the
best validation loss. TensorBoard logs go to `tensorboard/<timestamp>/`.
Training stops early after `patience` epochs without improvement in
validation loss.

## Inference

```sh
python predict.py -c config.yml -i input.wav -o predicted.wav -m checkpoint/20180208_235128/model_000031.h5
```

No pre-trained weights are included in this repository; replace the
checkpoint path with one produced by training.

`input.wav` is expected to be 48 kHz / mono / 16-bit PCM; the format is not
validated. The result is always written as 48 kHz / mono / 16-bit PCM.

> Note: the original `predict.py` pads the input incorrectly for some input
> lengths. Depending on the length, the output may be a few samples shorter
> than the input, or the script may fail on an empty window array.

## Legacy Version

To inspect the original February 2018 implementation exactly as it was
released:

```sh
git checkout 2018-original
```

## License

MIT License. See [LICENSE](LICENSE).
