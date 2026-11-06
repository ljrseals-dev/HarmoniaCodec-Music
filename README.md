# HarmoniaSynth — F0-conditioned singing voice head for HarmoniaCodec

**Status: scaffold only.**  
The underlying HarmoniaCodec ships **no trained weights**.  
This synthesis head must be trained on **your own singing** (or voices for which you hold written rights). Until that happens it reports the fingerprint `harmonia_synth_untrained` and any evaluation harness **must refuse** to score it as a finished model.

---

## What it does

HarmoniaSynth predicts the codec’s continuous 64-channel latent `z` at 120 frames per second from four frame-level streams:

| Stream   | Shape   | Meaning                                      |
|----------|---------|----------------------------------------------|
| phon     | [B, T]  | Phoneme / note-phone IDs                     |
| f0_hz    | [B, T]  | Fundamental frequency in Hz                   |
| voiced   | [B, T]  | 1 = voiced, 0 = unvoiced / silence           |
| loud     | [B, T]  | Frame loudness (log-RMS or similar)          |

The predicted latent is fed straight into the codec’s own `SubBandDecoderNN`.  
The vector quantiser is completely bypassed — no discrete tokens, no language-model stage.

Teacher signal at training time:

```text
z = frozen AcousticEncoderNN(real_singing_waveform)
```

This keeps the synthesis head inside the same continuous latent space the codec already knows how to decode.

---

## Why this design is good for singing

- **Explicit F0 + voicing** — vibrato, portamento, falsetto and breathy onsets stay under direct control instead of being hoped-for emergent behaviour.
- **Separate loudness channel** — dynamics are not forced to ride on the F0 or phoneme embeddings.
- **Dilated residual stack** — receptive field of roughly one second at 120 Hz, enough for musical phrases without a huge transformer.
- **Causal option** — once `CausalTransposedConv1d` is fixed, the same weights can run streaming.
- **Optional speaker embedding** — multi-singer fine-tuning is one argument away.

---

## File layout (relevant parts)

```text
HarmoniaCodec/
├── harmonia_neural.py      # AcousticEncoderNN, SubBandDecoderNN,
│                           # CausalConv1d, CausalTransposedConv1d, …
├── harmonia_synth.py       # ← this head (optimised for singing)
├── test_causal_streaming.py
├── test_streaming_equivalence.py
└── README_SINGING.md       # this document
```

---

## Quick start (after you have torch + codec weights)

```python
from harmonia_synth import build_singing_synth, train_step
import torch

synth, encoder = build_singing_synth(causal=True, device="cuda")

# Load real codec weights into encoder + synth.decoder here
# (the repository currently ships none)

opt = torch.optim.AdamW(synth.parameters(), lr=2e-4)

for batch in dataloader:          # see “Data preparation” below
    loss = train_step(synth, encoder, batch, opt)
    print(f"loss {loss:.4f}")

# Only after a real training run on rights-cleared data:
synth.mark_trained()
print(synth.fingerprint)          # → "harmonia_synth"
```

---

## Data preparation (you must do this yourself)

You need a few hours of **clean, dry singing** of the target voice(s).

Required frame-level annotations at 120 Hz (HOP = 200 samples @ 24 kHz):

1. **Phonemes** — forced alignment (Montreal Forced Aligner, or any singing-aware aligner).  
   Use a singing-friendly inventory (ARPAbet + extra symbols for sustained vowels is usually enough).
2. **F0** — RMVPE or CREPE (or the pitch tracker you already trust).  
   Store continuous Hz; the model takes `log(f0)` only on voiced frames.
3. **Voiced flag** — 1 where F0 is reliable, 0 elsewhere.
4. **Loudness** — short-time log-RMS or a simple LUFS-style estimator, also at 120 Hz.

A minimal extraction sketch (you still have to implement the aligner and F0 tracker):

```python
# pseudocode — replace with your real tools
wav, sr = load("singer.wav")          # must be 24 kHz mono
assert sr == 24000

phon  = align_phonemes(wav, lyrics)   # [T] int64
f0    = rmvpe(wav)                    # [T] float32 Hz
voiced = (f0 > 0).astype(np.float32)
loud  = log_rms(wav, hop=200)         # [T] float32

# save as torch tensors or a simple .pt / .npy pack
```

**Legal note:** only use recordings of voices for which you have explicit written permission (your own voice, licensed session singers, etc.). Do not scrape commercial or copyrighted material.

---

## Training recipe (practical)

```text
GPU          : 1× 12 GB is enough for the 256-dim residual stack
Batch size   : 8–16 (≈ 4–8 s of audio each)
Steps        : 50 k–150 k depending on data volume
LR           : 2e–4 AdamW, cosine decay after 10 k steps
Loss         : L1 on latent (primary) + optional loudness auxiliary
Freeze       : AcousticEncoderNN completely
Checkpoint   : keep the best latent L1 on a held-out song
```

After training call:

```python
synth.mark_trained(fingerprint="harmonia_synth")
torch.save(synth.state_dict(), "harmonia_synth.pt")
```

---

## Inference

```python
synth.eval()
with torch.no_grad():
    audio = synth(phon, f0_hz, voiced, loud)   # [1, 1, N]
```

For streaming (once the transposed-conv path is trustworthy):

```python
synth.reset()
for chunk in streaming_source:
    # chunk contains the four frame-level tensors for a few frames
    audio_chunk = synth(**chunk)
    play(audio_chunk)
```

---

## Causal / streaming status

| Component                | Status                                      |
|--------------------------|---------------------------------------------|
| CausalConv1d             | Fixed & verified (counter-based)            |
| CausalTransposedConv1d   | Older implementation — still fails tests    |
| SubBandDecoderNN causal  | Depends on the transposed conv              |
| HarmoniaSynth causal     | Ready on the predictor side; decoder blocked|

Do **not** claim real-time performance until:

```bash
python test_streaming_equivalence.py
python test_causal_streaming.py
```

both report zero failures for `CausalTransposedConv1d` and `SubBandDecoderNN`.

---

## Fingerprint & harness contract

```python
@property
def fingerprint(self) -> str:
    return "harmonia_synth" if self.trained else "harmonia_synth_untrained"
```

Any scoring or leaderboard system **must**:

1. Read `model.fingerprint`.
2. Refuse to accept or rank a model whose fingerprint is `harmonia_synth_untrained`.
3. Only accept `harmonia_synth` after the user has called `mark_trained()` on rights-cleared data.

This is intentional — the scaffold must never be mistaken for a finished singing-voice model.

---

## What is still missing

- Trained codec weights (encoder + decoder).
- A battle-tested `CausalTransposedConv1d` that passes the two streaming tests.
- The actual data-prep scripts (aligner + RMVPE/CREPE + loudness) — left to the user so that rights stay clear.
- Any listening-test results or MOS numbers (none exist yet).

---

## Licence & responsible use

This file inherits the licence of the parent HarmoniaCodec repository.  
Use only on voices you have the legal right to model.  
The authors of the scaffold accept no responsibility for misuse.

---

*HarmoniaSynth is a research scaffold, not a product.*  
Train it on your own voice, keep the fingerprint honest, and the rest follows.
