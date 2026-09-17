"""Transformer LM backbone with MoE blocks and optional vision injection.

Structure per block:  pre-norm attention -> pre-norm MoE.
Decoder-only (causal) with a KV cache for fast autoregressive sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.config import ModelConfig
from forge.model.moe import MoELayer, get_balance_loss
from forge.model.vision import PatchVisionEncoder


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)

        def split_heads(z):
            return z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        new_cache = None
        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k.detach(), v.detach())

        total = k.shape[2]
        # With cache the query attends over all past keys, so the mask only
        # needs to hide future positions relative to each query.
        offset = total - t
        mask = torch.triu(
            torch.full((t, total), float("-inf"), device=x.device), diagonal=offset + 1
        )
        att = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5 + mask
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out), new_cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.moe = MoELayer(
            cfg.dim,
            cfg.moe.num_experts,
            cfg.moe.top_k,
            cfg.moe.expert_hidden,
            dropout=cfg.dropout,
            jitter=cfg.moe.router_jitter,
            shared_experts=cfg.moe.shared_experts,
            shared_hidden=cfg.moe.shared_hidden,
            routing=cfg.moe.routing,
            capacity_factor=cfg.moe.capacity_factor,
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        a, new_cache = self.attn(h, cache)
        x = x + a
        x = x + self.moe(self.norm2(x))
        return x, new_cache


@dataclass
class LMOutput:
    logits: torch.Tensor
    moe_loss: torch.Tensor


class ForgeLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.dim))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.vision = (
            PatchVisionEncoder(cfg.vision, cfg.dim, cfg.dropout)
            if cfg.vision.enabled else None
        )
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def moe_layers(self) -> list[MoELayer]:
        return [b.moe for b in self.blocks]

    def forward(
        self,
        idx: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max {self.cfg.max_seq_len}")

        x = self.drop(self.tok_embed(idx))
        # Position ids must account for the KV cache length.
        past = cache[0][0].shape[2] if cache and cache[0] is not None else 0
        pos = self.pos_embed[:, past:past + t, :]
        x = x + pos

        if images is not None and self.vision is not None:
            vis = self.vision(images)                     # (B, summary_tokens, dim)
            x = torch.cat([vis, x], dim=1)                # prepend visual prefix

        new_cache = []
        for i, blk in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, c = blk(x, layer_cache)
            new_cache.append(c)

        if images is not None and self.vision is not None and targets is not None:
            # Targets only cover text tokens; visual prefix positions are dropped.
            x = x[:, self.cfg.vision.summary_tokens:, :]

        x = self.norm_f(x)
        logits = self.lm_head(x)
        moe_loss = get_balance_loss(self.moe_layers())
        return logits, moe_loss, new_cache

    def loss(self, logits, targets, moe_loss) -> tuple[torch.Tensor, torch.Tensor]:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
        )
        total = ce + self.cfg.moe.aux_loss_coef * moe_loss
        return total, ce

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.9,
        top_k: Optional[int] = 40,
        images: Optional[torch.Tensor] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        cache: Optional[list] = [None] * len(self.blocks)
        cur = idx
        first = True
        for _ in range(max_new_tokens):
            img = images if first else None
            logits, _, cache = self.forward(cur, images=img, cache=cache)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            cur = nxt
            idx = torch.cat([idx, nxt], dim=1)
            first = False
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx