"""Minimal vision tower: patch embed -> transformer encoder -> summary tokens.

This is the "VLM" half of the framework, kept small enough to train on CPU.
Images are split into non-overlapping patches, embedded with a linear layer,
plus a learned positional embedding, then run through a few encoder blocks.
The pooled output is projected back to the LM embedding width as a handful of
summary tokens, which the language model then attends over normally.

Scope note: this produces coarse visual conditioning, sufficient for tasks
like shape/brightness/pattern discrimination.  It is not a general-purpose
image understanding model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.config import VisionConfig


class PatchVisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig, target_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        self.patch_dim = p * p * cfg.channels
        self.n_patches = (cfg.image_size // p) ** 2

        self.patch_embed = nn.Linear(self.patch_dim, target_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, target_dim))
        self.norm = nn.LayerNorm(target_dim)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(target_dim, dropout=dropout) for _ in range(2)]
        )
        self.summary = nn.Parameter(torch.randn(1, cfg.summary_tokens, target_dim) * 0.02)
        self.proj = nn.Linear(target_dim, target_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) -> (B, summary_tokens, target_dim)."""
        b = images.shape[0]
        p = self.cfg.patch_size
        # (B, C, H, W) -> (B, n_patches, patch_dim)
        x = images.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, self.patch_dim)

        x = self.patch_embed(x) + self.pos
        for blk in self.blocks:
            x = blk(x)

        # Pool patches into summary tokens via a learned attention pooling.
        q = self.summary.expand(b, -1, -1)
        attn = torch.softmax(q @ x.transpose(1, 2) / x.shape[-1] ** 0.5, dim=-1)
        pooled = attn @ x
        return self.proj(self.norm(pooled))


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))