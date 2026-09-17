"""Sparse Mixture-of-Experts feed-forward layer.

A dense MLP is replaced by N small experts plus a linear router.  Only top_k
experts run per token, so FLOPs per token stay near that of a single expert
while total parameters grow with N -- the mechanism behind Mixtral and
DeepSeek-MoE, at a scale a CPU can actually train.

Design choices that matter:

1. **Shared experts** (``shared_experts``): always-on experts every token
   passes through.  They absorb the common patterns so the routed experts can
   specialise instead of all learning the same generic features.
2. **Router jitter** during training: without it a few experts win early and
   the rest never recover (expert collapse).
3. **Load-balancing aux loss**: required, not optional.  Top-k routing alone
   does not produce balanced traffic.
4. **Two routing modes**:
   * ``token_choice`` -- each token picks its top_k experts.  Standard, and
     the only mode safe for autoregressive decoding.
   * ``expert_choice`` -- each expert picks its own top-C tokens.  Gives a
     perfectly balanced assignment, but the selection sees the whole batch at
     once, so it is **non-causal** and must not be used for generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """SwiGLU expert: two up-projections gated together, one down-projection."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(dim, hidden, bias=False)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.w_gate(x)) * self.w_in(x)
        return self.drop(self.w_out(h))


@dataclass
class RoutingInfo:
    """Diagnostics for one MoE forward pass."""
    probs: torch.Tensor          # mean router probability per expert
    load: torch.Tensor           # fraction of tokens dispatched per expert
    dropped: float               # fraction dropped by the capacity limit


class MoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden: int,
        dropout: float = 0.0,
        jitter: float = 0.0,
        shared_experts: int = 0,
        shared_hidden: int = 0,
        routing: str = "token_choice",
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        assert top_k <= num_experts, "top_k cannot exceed num_experts"
        assert routing in ("token_choice", "expert_choice"), f"bad routing {routing!r}"
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.routing = routing
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(dim, expert_hidden, dropout) for _ in range(num_experts)
        )
        self.shared = nn.ModuleList(
            Expert(dim, shared_hidden or expert_hidden, dropout)
            for _ in range(shared_experts)
        )
        self.last_info: RoutingInfo | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        out = torch.zeros_like(flat)
        for expert in self.shared:
            out = out + expert(flat)

        logits = self.router(flat)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter
        probs = F.softmax(logits, dim=-1)

        if self.routing == "expert_choice":
            out = out + self._expert_choice(flat, probs)
        else:
            out = out + self._token_choice(flat, probs)

        return out.reshape(b, t, d)

    # ------------------------------------------------------------------
    def _token_choice(self, flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        load = torch.zeros(self.num_experts)
        for slot in range(self.top_k):
            idx = top_i[:, slot]
            w = top_w[:, slot].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                count = int(mask.sum())
                if count:
                    out[mask] += w[mask] * self.experts[e](flat[mask])
                load[e] += count

        self.last_info = RoutingInfo(
            probs=probs.detach().mean(dim=0),
            load=load / max(flat.shape[0], 1),
            dropped=0.0,
        )
        return out

    def _expert_choice(self, flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Each expert keeps its top-C tokens.  Non-causal -- training only."""
        n_tokens = flat.shape[0]
        capacity = max(1, int(self.capacity_factor * n_tokens * self.top_k
                              / self.num_experts))
        out = torch.zeros_like(flat)
        load = torch.zeros(self.num_experts)
        dropped = 0

        gate = probs.t()                                   # (E, N)
        top_w, top_i = torch.topk(gate, min(capacity, n_tokens), dim=-1)

        for e in range(self.num_experts):
            tok = top_i[e]
            w = top_w[e]
            assign = torch.zeros(n_tokens, device=flat.device)
            assign[tok] = w
            keep = assign > 0
            if keep.any():
                out[keep] += assign[keep].unsqueeze(-1) * self.experts[e](flat[keep])
            load[e] = int(keep.sum())
            dropped += n_tokens - int(keep.sum())

        total_slots = n_tokens * self.top_k
        self.last_info = RoutingInfo(
            probs=probs.detach().mean(dim=0),
            load=load / max(n_tokens, 1),
            dropped=min(1.0, dropped / max(total_slots, 1)),
        )
        return out

    # ------------------------------------------------------------------
    def has_routed(self) -> bool:
        return self.last_info is not None

    def routing_probs(self) -> torch.Tensor:
        if self.last_info is None:
            return torch.full((self.num_experts,), 1.0 / self.num_experts)
        return self.last_info.probs

    def routing_stats(self) -> torch.Tensor:
        """Mean router probability per expert from the last forward pass."""
        return self.routing_probs()

    def load_stats(self) -> torch.Tensor:
        if self.last_info is None:
            return torch.full((self.num_experts,), 1.0 / self.num_experts)
        return self.last_info.load


def get_balance_loss(moe_layers: list[MoELayer]) -> torch.Tensor:
    """Switch-Transformer style load-balancing loss.

        L = N * sum_i ( f_i * P_i )

    where ``f_i`` is the fraction of tokens *actually dispatched* to expert i
    and ``P_i`` the mean router probability for that expert.

    Two deliberate choices:

    * ``f_i`` comes from real dispatch counts, not probabilities.  A router can
      be extremely confident and still perfectly balanced; penalising that
      spends gradient on a non-problem.
    * Multiplying by ``N`` keeps the scale independent of expert count, so one
      coefficient works whether you have 4 experts or 64.
    """
    if not moe_layers:
        return torch.zeros(())
    total = torch.zeros(())
    counted = 0
    for layer in moe_layers:
        if not layer.has_routed():
            continue
        n = layer.num_experts
        load = layer.load_stats()
        f = load / (load.sum() + 1e-9)          # normalised dispatch fraction
        p = layer.routing_probs()
        p = p / (p.sum() + 1e-9)
        total = total + n * (f * p).sum()
        counted += 1
    return total / max(counted, 1)