"""harmonia_synth.py — F0-conditioned synthesis head for HarmoniaCodec (SCAFFOLD).

Idea: predict the codec's continuous latent z (64 ch, 24 kHz / 200 = 120 frames/s)
from frame-level phonemes + log-F0 + voicing + loudness, then render with the
codec's own SubBandDecoderNN. Bypasses the VQ, so no token modelling is needed.
Teacher signal at train time: z = frozen AcousticEncoderNN(real singing audio).

STATUS (be honest): written without torch available in the authoring sandbox.
It is syntax-checked only, NOT run. The codec itself has no trained weights, and
this head needs training on YOUR OWN voice (or voices you hold written rights to).
Until trained it must report fingerprint "harmonia_synth_untrained" and the
harness must refuse to score it as "harmonia_synth".
"""
import torch, torch.nn as nn, torch.nn.functional as F
from harmonia_neural import SubBandDecoderNN, AcousticEncoderNN, CausalConv1d  # same repo

SR, HOP = 24000, 200            # 200 = prod(strides) of the codec -> 120 frames/s
FRAME_RATE = SR // HOP

class HarmoniaSynth(nn.Module):
    def __init__(self, n_phonemes=128, latent=64, dim=256, layers=6, causal=True):
        super().__init__()
        self.causal = causal
        self.ph = nn.Embedding(n_phonemes, 64)
        self.cond = nn.Conv1d(64 + 3, dim, 1)               # +logf0, voiced, loudness
        mk = (lambda ci, co, k, d: CausalConv1d(ci, co, k, dilation=d)) if causal else \
             (lambda ci, co, k, d: nn.Conv1d(ci, co, k, padding=d * (k // 2), dilation=d))
        self.blocks = nn.ModuleList([mk(dim, dim, 5, 2 ** (i % 4)) for i in range(layers)])
        self.out = nn.Conv1d(dim, latent, 1)
        self.decoder = SubBandDecoderNN(latent=latent, dim=128, causal=causal)  # load codec weights
        self.trained = False                                 # flip only after real training

    @property
    def fingerprint(self):
        return "harmonia_synth" if self.trained else "harmonia_synth_untrained"

    def predict_latent(self, phon, f0_hz, voiced, loud):
        """all inputs [B,T] at FRAME_RATE; returns z_hat [B,64,T]"""
        lf0 = torch.where(voiced > 0, torch.log(f0_hz.clamp(min=20.0)), torch.zeros_like(f0_hz))
        x = torch.cat([self.ph(phon).transpose(1, 2), lf0[:, None], voiced[:, None].float(), loud[:, None]], 1)
        h = self.cond(x)
        for b in self.blocks:
            h = h + F.silu(b(h))
        return self.out(h)

    def forward(self, phon, f0_hz, voiced, loud):
        return self.decoder(self.predict_latent(phon, f0_hz, voiced, loud))   # [B,1,T*HOP]

def train_step(model, encoder, batch, opt, w_latent=1.0):
    """batch: wav [B,1,N] real singing, plus frame-level phon/f0/voiced/loud [B,T]."""
    with torch.no_grad():
        z = encoder(batch["wav"])                                         # teacher latent
    z_hat = model.predict_latent(batch["phon"], batch["f0"], batch["voiced"], batch["loud"])
    T = min(z.shape[-1], z_hat.shape[-1])
    loss = w_latent * F.l1_loss(z_hat[..., :T], z[..., :T])
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)
