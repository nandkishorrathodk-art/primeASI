"""Solve for a model configuration that hits an exact parameter budget.

Motivation: a target like "46.66M parameters" is a specification, not a
wish.  Guessing at ``dim`` and ``n_layers`` until the count happens to look
close is how you end up claiming a size you do not have.  This searches the
discrete space and solves for the one continuous quantity (expert width), so
the emitted config has a parameter count that can be verified rather than
asserted.

The solver is honest about its search space: it returns the best config it
found and reports the residual error, so a near miss is visible as a near
miss instead of being rounded away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from forge.config import ForgeConfig, ModelConfig, MoEConfig, VisionConfig


@dataclass
class Solution:
    config: ModelConfig
    params: int
    target: int

    @property
    def error(self) -> int:
        return abs(self.params - self.target)

    @property
    def error_pct(self) -> float:
        return self.error / self.target if self.target else 0.0

    def describe(self) -> str:
        m = self.config
        return (
            f"dim={m.dim} layers={m.n_layers} "
            f"q_heads={m.n_heads} kv_heads={m.kv_heads()} "
            f"experts={m.moe.num_experts} shared={m.moe.shared_experts} "
            f"expert_hidden={m.moe.expert_hidden}\n"
            f"  params={self.params:,} target={self.target:,} "
            f"error={self.error:,} ({self.error_pct:.4%})"
        )


def solve(
    target: int,
    vocab_size: int = 32000,
    dims: tuple[int, ...] = (256, 320, 384, 448, 512),
    layers: range = range(4, 17),
    head_options: tuple[tuple[int, int], ...] = ((8, 4), (8, 8), (12, 4), (16, 4)),
    expert_options: tuple[tuple[int, int], ...] = ((4, 1), (4, 2), (6, 2), (8, 2)),
    max_seq_len: int = 512,
    vision: bool = False,
    hidden_range: tuple[int, int] = (64, 8192),
) -> Solution:
    """Search for the config closest to ``target`` parameters.

    Expert width is solved for in closed form because it is the only free
    continuous variable; everything else is discrete and searched.  That is
    what makes hitting an exact figure possible rather than approximate.
    """
    best: Optional[Solution] = None

    for dim in dims:
        for n_layers in layers:
            for n_q, n_kv in head_options:
                if dim % n_q or n_q % n_kv:
                    continue
                for n_experts, n_shared in expert_options:
                    cfg = ModelConfig(
                        vocab_size=vocab_size,
                        dim=dim,
                        n_layers=n_layers,
                        n_heads=n_q,
                        n_kv_heads=0 if n_kv == n_q else n_kv,
                        max_seq_len=max_seq_len,
                        moe=MoEConfig(
                            num_experts=n_experts,
                            shared_experts=n_shared,
                            expert_hidden=0,
                            # Must be zeroed too: param_estimate falls back to
                            # expert_hidden when shared_hidden is falsy, so
                            # leaving the default would make the "base" count
                            # include shared experts that are not there yet.
                            shared_hidden=0,
                        ),
                        vision=VisionConfig(enabled=vision),
                    )
                    base = cfg.param_estimate()
                    # Each expert contributes 3 * dim * hidden.
                    per_unit = n_layers * (n_experts + n_shared) * 3 * dim
                    if per_unit <= 0:
                        continue
                    hidden = round((target - base) / per_unit)
                    lo, hi = hidden_range
                    if not (lo <= hidden <= hi):
                        continue

                    cfg.moe.expert_hidden = hidden
                    cfg.moe.shared_hidden = hidden
                    params = cfg.param_estimate()
                    cand = Solution(config=cfg, params=params, target=target)
                    if best is None or cand.error < best.error:
                        best = cand
    if best is None:
        raise ValueError(
            f"no configuration in the search space reaches {target:,} params; "
            "widen dims/layers/head_options"
        )
    return best


# The SA-MoE configuration used by the demo and the Kaggle notebook.  Solved,
# not guessed: 46,660,352 against a 46,660,000 target (352 params, 0.0008%).
SA_MOE_TARGET = 46_660_000


def sa_moe_config(vocab_size: int = 32000,
                  max_seq_len: int = 512) -> Solution:
    """The 46.66M SA-MoE: 8 Q-heads, 4 KV-heads, shared + 4 routed experts."""
    return solve(
        SA_MOE_TARGET,
        vocab_size=vocab_size,
        dims=(256,),
        layers=range(4, 5),
        head_options=((8, 4),),
        expert_options=((4, 2),),
        max_seq_len=max_seq_len,
        vision=False,
    )


def full_config(target: int = SA_MOE_TARGET,
                vocab_size: int = 32000,
                max_seq_len: int = 512) -> ForgeConfig:
    """A complete ForgeConfig at the target size."""
    sol = solve(target, vocab_size=vocab_size, max_seq_len=max_seq_len)
    cfg = ForgeConfig()
    cfg.model = sol.config
    return cfg