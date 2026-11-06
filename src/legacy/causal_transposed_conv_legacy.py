"""ARCHIVAL ONLY: original failing CausalTransposedConv1d implementation.

This file preserves the pre-overlap-add implementation from
harmonia_neural.py for forensic comparison and regression analysis.

Do not import this class in production or use it as the repaired streaming
decoder. It replays cached latent frames and trims a fixed right-side amount;
the implementation can re-emit overlap and produce incorrect chunk lengths.
The active candidate implementation remains in harmonia_neural.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CausalTransposedConv1d(nn.Module):
    """Original implementation retained verbatim for historical reference."""

    def __init__(self, in_ch, out_ch, kernel, stride, dilation=1, groups=1):
        super().__init__()
        self.kernel, self.stride, self.dilation = kernel, stride, dilation
        self.pad = (kernel - 1) * dilation
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride,
                                       dilation=dilation, groups=groups)
        self._state: Optional[torch.Tensor] = None
        self._state_len = max(0, (self.kernel - 1 + stride - 1) // stride)

    def reset(self): self._state = None

    def forward(self, x):
        if self._state is not None:
            x = torch.cat([self._state, x], dim=-1)
        y = self.conv(x)
        if self.pad:
            y = y[..., :-self.pad]
        if self._state_len > 0:
            self._state = x[..., -self._state_len:].detach()
        return y
