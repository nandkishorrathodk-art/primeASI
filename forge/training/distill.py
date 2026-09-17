"""Lightweight self-distillation.

Why not "train five frontier models into your own model": you cannot.  Model
weights are not published for GPT/Claude/Gemini/Grok, and even if you had
them, absorbing them requires datacentre-scale compute.  What is actually
possible, and what this module does:

  * **Self-distillation / EMA teacher** -- keep an exponential moving average
    of the student and distil it back into the student.  Real, cheap, and it
    measurably smooths the loss on small runs.
  * **Soft-target distillation from any teacher you legitimately have** -- if
    you hold API keys, teachers can supply *soft labels* (token
    distributions).  That is legitimate training signal, not weight absorption.

Anything claiming to "learn from" a closed model without either an API or
weights is marketing.
"""
from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from forge.model.transformer import ForgeLM


class EMATeacher:
    """Exponential moving average of the student model.

    Acts as a free, always-available teacher so distillation has a real signal
    with no external dependencies.
    """

    def __init__(self, student: ForgeLM, decay: float = 0.995) -> None:
        self.decay = decay
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: ForgeLM) -> None:
        d = self.decay
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.mul_(d).add_(sp.detach(), alpha=1 - d)
        for tb, sb in zip(self.teacher.buffers(), student.buffers()):
            tb.copy_(sb)

    def soft_targets(self, idx: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits, _, _ = self.teacher(idx)
        return logits / temperature


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL to the teacher plus hard-label cross-entropy.

    Returns (total, kl, ce).
    """
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    hard = F.cross_entropy(s, targets.reshape(-1), ignore_index=-100)
    kl = F.kl_div(
        F.log_softmax(s / temperature, dim=-1),
        F.softmax(t / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * kl + (1 - alpha) * hard, kl, hard


@dataclass
class TeacherCache:
    """Cached teacher responses, so API cost is paid once per prompt."""
    path: str = "runs/distill_cache/teacher.pkl"

    def load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                return pickle.load(fh)
        return {}

    def save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as fh:
            pickle.dump(data, fh)


def collect_teacher_texts(
    prompts: list[str],
    providers: Optional[list[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> dict[str, str]:
    """Query any configured providers and return a prompt -> completion map.

    Providers without credentials are skipped, never simulated.
    """
    from forge.agents.backends import RemoteBackend, available_providers

    live = [p for p in (providers or available_providers()) if p in available_providers()]
    out: dict[str, str] = {}
    for prompt in prompts:
        for provider in live:
            try:
                backend = RemoteBackend(provider)
                out[f"{provider}::{prompt}"] = backend.complete(
                    "Continue the text.", prompt, max_tokens=max_tokens
                )
            except RuntimeError:
                continue
    return out