# HarmoniaCodec Music

<p align="center">
  <img src="img/Music.png" alt="Music" width="300">
</p>

## A research-grade singing-voice synthesis and audio-codec project

**HarmoniaCodec Music explores a codec-conditioned path from phonetic and expressive controls to synthesized singing audio.** It combines a NumPy reference codec, an optional PyTorch training stack, an F0-conditioned synthesis head, and a verification-first development workflow.

> **Research status — not a production streaming service.** This repository currently ships no trained codec or synthesis weights, no listening-test results, and no production-ready streaming decoder. The causal decoder is explicitly blocked by streaming tests. Treat the project as an active research scaffold, not a finished commercial music platform.

[Core codec README](README.md) · [Music synthesis overview](README_MUSIC.md) · [Training guide](TRAINING.md) · [Model status](MODEL_STATUS.json) · [Responsible use](RESPONSIBLE_USE.md)

---

## The science in one view

```text
Rights-cleared singing audio
        │
        ├── phoneme alignment ──────┐
        ├── F0 / voicing analysis ──┤  external tools / weights
        └── loudness features ──────┘
                                    │  120 frames / second
                                    ▼
                         HarmoniaSynth prediction head
                                    │
                              latent z [64, T]
                                    │
                                    ▼
                          SubBandDecoder / codec
                                    │
                              24 kHz waveform
```

The music path conditions synthesis on phonetic content, fundamental frequency, voicing, and loudness. The synthesis head predicts a continuous latent representation; the codec decoder renders the waveform. The pipeline and its current dependencies are described in [README_MUSIC.md](README_MUSIC.md).

## What is in the repository

| Area | Purpose | Current evidence / status |
|---|---|---|
| NumPy reference codec | Residual vector quantization, variable-rate stage allocation, lossless escape path | Self-test coverage is documented in the core README; random initialization is not a quality-trained model |
| PyTorch codec stack | Trainable encoder/decoder and supporting neural components | Training path exists; real training and perceptual quality are not established |
| Singing synthesis head | Predict codec latents from phoneme, F0, voicing, and loudness features | Requires training; no trained weights ship |
| Alignment and F0 | Produce phoneme and pitch features | External dependencies; not bundled |
| Streaming decoder | Causal/chunked rendering | **Blocked:** causal transposed-convolution streaming fails its tests |
| Benchmark harness | Compare supported classical codecs using real local encoder subprocesses | Classical codec measurements are documented; HarmoniaCodec is not yet a trained competitor row |

Run `python tools/model_status.py` for the machine-readable component status summarized for humans. Presence of source code or a checkpoint file alone is not proof of model quality.

## Music and audio delivery formats

The following are **industry delivery and playback categories**, not a claim that HarmoniaCodec currently implements every format or streaming protocol.

| Category | Typical use | Engineering considerations |
|---|---|---|
| Lossless file delivery | FLAC, ALAC, WAV | Archival, mastering handoff, offline libraries; larger payloads |
| Lossy music delivery | MP3, AAC-LC, Ogg Vorbis, Opus | Bandwidth-efficient playback; quality depends on encoder, bitrate, and source |
| Adaptive HTTP streaming | HLS, MPEG-DASH | Segmented delivery with bitrate adaptation; requires packaging, manifests, CDN/origin, and player support |
| Low-latency interactive audio | WebRTC / RTP | Real-time conversational or collaborative audio; latency, jitter buffers, packet loss, and clock drift matter |
| Live broadcast / radio | Icecast / SHOUTcast-style workflows | Continuous streams, metadata, reconnect behavior, and operational monitoring |
| Download-and-play / offline | Local files and cached segments | Explicit storage policy, integrity checks, and rights-aware offline access |
| Studio / production interchange | PCM WAV, AIFF, stems | Predictable sample rates, bit depth, channel layouts, and synchronization |

**Codec research is not the same as a streaming platform.** HLS/DASH packaging, transport, CDN delivery, authentication, subscriptions, catalog licensing, and a production player are separate system layers and are not represented as shipped features here.

## Supported codec families in the benchmark harness

The existing classical benchmark path documents FLAC, ALAC, WavPack, MP3, AAC-LC, Vorbis, and Opus via local FFmpeg subprocesses. Optional DAC and Encodec adapters are unverified and may skip when their packages or weights are unavailable. HarmoniaCodec itself is not yet included as a measured competitor because a trained checkpoint and adapter are not present.

No competitor-superiority claim is made. No PESQ, STOI, ViSQOL, MUSHRA, or UTMOS result is claimed.

## Quick start

Python 3.9+.

```bash
pip install numpy scipy soundfile
python harmonia.py selftest --expanded
python harmonia_signalgen.py
python tools/model_status.py
```

Optional neural development:

```bash
pip install torch torchaudio
python harmonia_neural.py train --dry-run
```

The dry-run is a pipeline smoke exercise, not evidence of convergence or perceptual quality. Read [TRAINING.md](TRAINING.md) before attempting a real training run. The guide specifies rights-cleared data, external feature extraction, training order, and acceptance gates.

## Research and validation roadmap

1. **Resolve codec training blockers** — fix and test the known neural loss-shape and missing-WavLM guard issues before a real training run.
2. **Train and attest the codec** — use a documented, rights-cleared manifest; retain evaluation artifacts and human listening review.
3. **Train the synthesis head** — evaluate on held-out speakers/recordings and compare against the zero-latent baseline.
4. **Repair causal decoding** — make chunked output equivalent to single-pass output within an explicit tolerance; test boundaries, state reset, and long-duration drift.
5. **Establish perceptual evidence** — publish reproducible objective metrics and controlled listening-test methodology before making quality claims.
6. **Define a versioned bitstream** — specify framing, metadata, error handling, compatibility, and security before claiming interoperable encoded files.
7. **Only then explore streaming delivery** — separately design HLS/DASH or low-latency transport, player integration, buffering, and operational deployment.

## Responsible data and voice use

Train only on recordings you own or for which you hold documented written permission. Preserve the dataset manifest and consent evidence required by the project. Review [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) and [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md) before preparing training data.

A codec is not a consent system, identity-verification system, or legal clearance mechanism. External aligners, pitch trackers, and pretrained models have their own licenses and usage conditions; verify those before commercial deployment.

## Privacy and network behavior

Do not interpret “local-first” as “every optional workflow is offline.” The NumPy reference path and local classical FFmpeg benchmark can run locally. Optional semantic-model setup and optional DAC/Encodec integrations may retrieve external model weights. Review the network-behavior disclosure in the [core README](README.md) and configure optional components deliberately for air-gapped environments.

## Documentation map

- [README.md](README.md) — codec architecture, verified behavior, known issues, privacy/network disclosure
- [README_MUSIC.md](README_MUSIC.md) — singing synthesis pipeline and current limitations
- [TRAINING.md](TRAINING.md) — data requirements, training sequence, and acceptance gates
- [MODEL_STATUS.json](MODEL_STATUS.json) — machine-readable model/component status
- [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) — responsible voice-data guidance
- [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md) — dataset manifest requirements

## License

MIT. See [LICENSE](LICENSE) and retain the existing contributor notice.
