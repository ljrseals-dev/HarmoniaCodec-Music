"""harmonia_synth.py — F0-conditioned singing-voice synthesis head for HarmoniaCodec.

Predicts the codec's continuous latent z (64 ch @ 120 frames/s) from
frame-level phonemes + log-F0 + voicing + loudness, then renders with the
codec's own SubBandDecoderNN. Bypasses the VQ entirely — no token modelling.

Teacher at train time: z = frozen AcousticEncoderNN(real singing).

STATUS (honest):
  - Syntax-checked only in the authoring environment (no torch).
  - Codec itself ships no trained weights.
  - This head must be trained on YOUR OWN voice (or voices you hold written
    rights to). Until then it reports fingerprint "harmonia_synth_untrained"
    and any scoring harness must refuse to treat it as a finished model.
  - Causal mode depends on a correct CausalTransposedConv1d (still the older
    implementation in harmonia_neural.py). Do not trust streaming until the
    two streaming tests pass.

Optimizations for singing voice (vs. a generic speech head):
  - Explicit log-F0 + voiced flag (critical for vibrato, portamento, falsetto).
  - Separate loudness channel so dynamics and pitch are not entangled.
  - Residual dilated stack with singing-friendly receptive field (~1 s).
  - Optional speaker embedding slot for multi-singer fine-tuning later.
  - Causal path ready once the transposed-conv fix lands.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harmonia_neural import (
    SubBandDecoderNN,
    AcousticEncoderNN,
    CausalConv1d,
)

SR = 24_000
HOP = 200                       # prod(strides) of AcousticEncoderNN = 2·4·5·5
FRAME_RATE = SR // HOP          # 120 Hz


class HarmoniaSynth(nn.Module):
    """
    F0-conditioned continuous latent predictor + codec decoder.

    Inputs (all at FRAME_RATE):
        phon    : LongTensor [B, T]   phoneme / note-phone IDs
        f0_hz   : FloatTensor [B, T]  fundamental frequency in Hz
        voiced  : Float/Bool  [B, T]  1 = voiced, 0 = unvoiced / silence
        loud    : FloatTensor [B, T]  frame loudness (e.g. log RMS or LUFS-ish)

    Output:
        waveform [B, 1, T * HOP]
    """

    def __init__(
        self,
        n_phonemes: int = 128,
        latent: int = 64,
        dim: int = 256,
        layers: int = 6,
        causal: bool = True,
        speaker_dim: int = 0,          # 0 = single-singer; >0 enables multi-singer
    ):
        super().__init__()
        self.causal = causal
        self.latent = latent
        self.speaker_dim = speaker_dim

        # Phoneme embedding (singing-friendly inventory size)
        self.ph = nn.Embedding(n_phonemes, 64)

        # Conditioning: phoneme emb + log-F0 + voiced + loudness (+ optional speaker)
        cond_ch = 64 + 3 + (speaker_dim if speaker_dim > 0 else 0)
        self.cond = nn.Conv1d(cond_ch, dim, kernel_size=1)

        # Dilated residual stack — receptive field covers ~1 s of singing
        # dilations cycle 1,2,4,8 so the field grows quickly without huge kernels
        def make_block(ci: int, co: int, k: int, d: int) -> nn.Module:
            if causal:
                return CausalConv1d(ci, co, k, dilation=d)
            pad = d * (k // 2)
            return nn.Conv1d(ci, co, k, padding=pad, dilation=d)

        self.blocks = nn.ModuleList(
            [make_block(dim, dim, 5, 2 ** (i % 4)) for i in range(layers)]
        )
        self.out = nn.Conv1d(dim, latent, kernel_size=1)

        # Re-use the codec decoder (weights must be loaded separately)
        self.decoder = SubBandDecoderNN(latent=latent, dim=128, causal=causal)

        # Training bookkeeping — never claim a trained model by default
        self.trained = False
        self._fingerprint_override: Optional[str] = None

    @property
    def fingerprint(self) -> str:
        if self._fingerprint_override is not None:
            return self._fingerprint_override
        return "harmonia_synth" if self.trained else "harmonia_synth_untrained"

    def mark_trained(self, fingerprint: str = "harmonia_synth") -> None:
        """Call only after a real training run on rights-cleared data."""
        self.trained = True
        self._fingerprint_override = fingerprint

    def predict_latent(
        self,
        phon: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        loud: torch.Tensor,
        speaker: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns z_hat [B, latent, T] at FRAME_RATE.
        """
        # Safe log-F0: only on voiced frames, clamp to avoid log(0)
        lf0 = torch.where(
            voiced > 0.5,
            torch.log(f0_hz.clamp(min=20.0)),
            torch.zeros_like(f0_hz),
        )

        ph_emb = self.ph(phon).transpose(1, 2)          # [B, 64, T]
        x = torch.cat(
            [
                ph_emb,
                lf0[:, None, :],
                voiced[:, None, :].float(),
                loud[:, None, :],
            ],
            dim=1,
        )

        if self.speaker_dim > 0:
            if speaker is None:
                raise ValueError("speaker embedding required when speaker_dim > 0")
            # speaker: [B, speaker_dim] → broadcast over time
            sp = speaker[:, :, None].expand(-1, -1, x.shape[-1])
            x = torch.cat([x, sp], dim=1)

        h = self.cond(x)
        for block in self.blocks:
            h = h + F.silu(block(h))
        return self.out(h)

    def forward(
        self,
        phon: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        loud: torch.Tensor,
        speaker: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.predict_latent(phon, f0_hz, voiced, loud, speaker)
        return self.decoder(z)          # [B, 1, T * HOP]

    def reset(self) -> None:
        """Reset any causal state inside the decoder (and future causal blocks)."""
        for m in self.modules():
            if hasattr(m, "reset") and m is not self:
                m.reset()


def train_step(
    model: HarmoniaSynth,
    encoder: AcousticEncoderNN,
    batch: dict,
    opt: torch.optim.Optimizer,
    w_latent: float = 1.0,
    w_loud: float = 0.0,
) -> float:
    """
    One optimisation step.

    batch keys:
        wav     [B, 1, N]   real singing (teacher)
        phon    [B, T]
        f0      [B, T]
        voiced  [B, T]
        loud    [B, T]
        speaker [B, D]      optional
    """
    encoder.eval()
    with torch.no_grad():
        z = encoder(batch["wav"])

    z_hat = model.predict_latent(
        batch["phon"],
        batch["f0"],
        batch["voiced"],
        batch["loud"],
        batch.get("speaker"),
    )

    T = min(z.shape[-1], z_hat.shape[-1])
    loss = w_latent * F.l1_loss(z_hat[..., :T], z[..., :T])

    # Optional auxiliary: encourage the predicted latent to carry loudness
    # (useful when the decoder is already trained and we want dynamics)
    if w_loud > 0.0 and "loud" in batch:
        # crude proxy: mean absolute latent energy vs. target loudness
        energy = z_hat[..., :T].abs().mean(dim=1)
        loss = loss + w_loud * F.l1_loss(energy, batch["loud"][..., :T])

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# Convenience: freeze the codec encoder and attach a fresh synth head
# ---------------------------------------------------------------------------

def build_singing_synth(
    n_phonemes: int = 128,
    causal: bool = True,
    speaker_dim: int = 0,
    device: str = "cpu",
) -> tuple[HarmoniaSynth, AcousticEncoderNN]:
    """
    Returns (synth, frozen_encoder).

    The encoder is put into eval mode and all its parameters are frozen.
    You still need to load real codec weights before training or inference
    will produce anything other than noise.
    """
    encoder = AcousticEncoderNN(dim=128, latent=64, strides=(2, 4, 5, 5), causal=causal)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    synth = HarmoniaSynth(
        n_phonemes=n_phonemes,
        latent=64,
        dim=256,
        layers=6,
        causal=causal,
        speaker_dim=speaker_dim,
    )
    return synth.to(device), encoder.to(device)
