"""Wiring the trained model in as a reasoning backend.

``backends.build_backend("local")`` returns ``None`` because the caller has to
supply the model.  This module is that caller, so the connection is not left
as an exercise for the user.

The honest caveat, stated here because it determines how the pipeline behaves:
a 2.6M-parameter byte-level model produces text that mostly will *not* parse
as a plan.  That is expected at this scale, not a bug.  The pipeline is built
so that an unparseable proposal degrades to a rules-based plan **visibly** --
recorded in the audit chain -- rather than silently or not at all.

This also means the model improves by growing the *plan-grammar* share of its
training corpus, not just by growing in parameters.
"""
from __future__ import annotations

import os
from typing import Optional

from forge.agents.backends import Backend, LocalBackend, RuleBackend


def local_backend_from_checkpoint(
    path: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
) -> LocalBackend:
    """Load a ForgeLM checkpoint and wrap it as a reasoning backend."""
    from forge.training.trainer import Trainer

    model, _cfg = Trainer.load(path)
    from forge.tokenizer import TOKENIZER

    return LocalBackend(
        model=model,
        tokenizer=TOKENIZER,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def resolve_backend(
    spec: str,
    checkpoint: Optional[str] = None,
    fallback: Optional[Backend] = None,
) -> Backend:
    """Resolve a backend spec, including 'local' with a checkpoint path.

    'local' without an existing checkpoint falls back rather than crashing: a
    missing file should not take down the whole loop.
    """
    if spec == "local":
        if checkpoint and os.path.exists(checkpoint):
            return local_backend_from_checkpoint(checkpoint)
        return fallback or RuleBackend()
    if spec == "rules":
        return RuleBackend()
    from forge.agents.backends import RemoteBackend, PROVIDERS

    if spec in PROVIDERS:
        return RemoteBackend(spec)
    raise ValueError(f"unsupported backend spec: {spec!r}")


def build_team_backends(
    spec: str,
    checkpoint: Optional[str] = None,
    per_domain: Optional[dict[str, str]] = None,
    checkpoint_for: Optional[dict[str, str]] = None,
) -> dict[str, Backend]:
    """Build a per-domain backend map.

    ``per_domain`` lets each sub-agent use a different provider (e.g. security
    on one model, vision on another) when the user holds those keys.
    ``checkpoint_for`` lets a domain use its own trained checkpoint.
    """
    from forge.agents.roles import DOMAINS

    checkpoint_for = checkpoint_for or {}
    backends: dict[str, Backend] = {}
    for domain in DOMAINS:
        domain_spec = (per_domain or {}).get(domain, spec)
        backends[domain] = resolve_backend(
            domain_spec, checkpoint=checkpoint_for.get(domain, checkpoint)
        )
    return backends