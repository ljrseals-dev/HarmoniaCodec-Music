# HarmoniaCodec Music

F0-conditioned singing-voice synthesis built on the [HarmoniaCodec](README.md) neural codec. A small head predicts the codec's continuous latent from phonemes, pitch, voicing and loudness; the codec's own decoder renders audio.

> **Status: research scaffold.** The repository ships **no trained weights** and no listening-test results. Run `python tools/model_status.py` to see exactly what is missing.

## What needs training (at a glance)

```text
$ python tools/model_status.py
* codec_ae       trainable  NEEDS_TRAINING     python train_codec_ae.py --manifest manifest.json --out checkpoints/codec ...
* synth_head     trainable  NEEDS_TRAINING     python train_synth.py --manifest manifest.json --codec checkpoints/codec ...
* aligner        external   MISSING_EXTERNAL   (forced aligner, not in this repo)
* f0_tracker     external   MISSING_EXTERNAL   (RMVPE or CREPE weights, not in this repo)
  causal_decoder code_fix   BLOCKED            streaming transposed-conv fails its tests
```

The registry is [`MODEL_STATUS.json`](MODEL_STATUS.json) (machine-readable). Step-by-step instructions: [TRAINING.md](TRAINING.md).

## Pipeline

```text
singing wav -> aligner (phonemes) + F0 tracker (f0, voiced) + loudness   @120 frames/s
            -> HarmoniaSynth (trained: train_synth.py)  -> latent z [64, T]
            -> SubBandDecoder (trained: train_codec_ae.py) -> 24 kHz waveform
```

| Component | In this repo | Needs |
|---|---|---|
| Codec encoder/decoder | code | training (`train_codec_ae.py`) |
| Synthesis head | code | training (`train_synth.py`) |
| Forced aligner, F0 tracker | no | external install |
| Streaming (causal) decoder | code, **broken** | code fix, then the two streaming tests |
| Training-claim gate | code + tests | `harmonia_gate.attest_trained()` |

## Quick start

```bash
pip install numpy scipy            # inference / checks
pip install torch torchaudio       # training
python harmonia.py selftest --expanded       # NumPy codec self-test
python -m unittest discover -s tests         # torch-free tests
python tools/model_status.py                 # what is trained / missing
```

## Honest limits

- Not verified here: any torch code path (training scripts are syntax-checked only until you run them).
- The codec autoencoder has no adversarial loss, so early reconstructions will sound muffled; judge by ear.
- Fingerprint `harmonia_synth_untrained` is reported until the gate in [`harmonia_gate.py`](harmonia_gate.py) attests a rights-cleared training run. Benchmark harnesses should refuse untrained fingerprints.

## Responsible use

Train only on your own voice or voices you hold **written** permission for. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) and [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).

## License

MIT. See [LICENSE](LICENSE) (includes the original HarmoniaCodec contributors' notice, which must be kept).
