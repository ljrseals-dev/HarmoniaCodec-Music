# Training guide

Start here: `python tools/model_status.py` prints every component, its status and the next command. Status comes from files on disk and each component's gate file; presence of weights is **not** proof of quality.

## Order of operations

| # | Component | Command | Produces | Acceptance gate |
|---|---|---|---|---|
| 0 | Data + rights manifest | see [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md) | `manifest.json` | every recording `own` or `written`, consent hash, speaker id |
| 1 | Aligner, F0 tracker (external) | install; produce `*.feat.npz` | phon/f0/voiced/loud @120 Hz | frame counts cover the audio |
| 2 | Codec autoencoder | `python train_codec_ae.py --manifest manifest.json --out checkpoints/codec --steps 100000` | `codec_encoder.pt`, `codec_decoder.pt`, `codec_report.json` | `improved_over_baseline: true` **and** a human listening check |
| 3 | Synthesis head | `python train_synth.py --manifest manifest.json --codec checkpoints/codec --out checkpoints/synth` | `harmonia_synth.pt`, `synth_eval.json`, `manifest.bound.json` | held-out latent L1 < zero-latent baseline |
| 4 | Attest | `harmonia_gate.attest_trained(model, ckpt, manifest, eval)` | audit line in `harmonia_audit.jsonl` | set `canary_pass: true` in `synth_eval.json` only after running your benchmark canaries |

Step 4 is the only sanctioned way to change the model fingerprint from `harmonia_synth_untrained` to `harmonia_synth`.

## Data guidance (rules of thumb, not measured results)

A few hours of clean, dry singing at 24 kHz mono 16-bit PCM; one GPU with roughly 12 GB; tens of thousands to ~150k steps. Convert audio with `ffmpeg -i in.wav -ac 1 -ar 24000 -sample_fmt s16 out.wav`.

## Known blockers

- `causal_decoder`: `CausalTransposedConv1d` fails the streaming tests; train and use the non-causal models until fixed. Causal and non-causal checkpoints are **not** interchangeable.
- `neural_codec_full`: README Known Issue #3 must be resolved before a real run.
- Weights licences for external tools (RMVPE, aligners, any pretrained vocoder) must be checked by you before commercial use.
