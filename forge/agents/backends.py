"""Pluggable agent backends.

An agent's reasoning can come from:

* ``RuleBackend``   - deterministic heuristics.  Always available, zero cost,
                      fully reproducible.  Used by tests.
* ``LocalBackend``  - the ForgeLM trained in this repo.  No API needed.
* ``RemoteBackend`` - any OpenAI-compatible chat-completions endpoint.

Important honesty note: a heterogenous "council of GPT/Claude/Gemini/Grok/..."
only exists for a user who actually holds those API keys.  This module detects
which providers are configured, uses them when present, and otherwise degrades
to the local model.  It never fabricates a provider's opinion.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

# OpenAI-compatible chat endpoints.  base_url can be overridden per provider.
PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"env": "OPENAI_API_KEY", "base": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "base": "https://api.anthropic.com/v1", "model": "claude-3-5-sonnet-20241022"},
    "google": {"env": "GOOGLE_API_KEY", "base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-1.5-flash"},
    "xai": {"env": "XAI_API_KEY", "base": "https://api.x.ai/v1", "model": "grok-2-latest"},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "nvidia": {"env": "NVIDIA_API_KEY", "base": "https://integrate.api.nvidia.com/v1", "model": "meta/llama-3.1-8b-instruct"},
}


def available_providers() -> list[str]:
    return [name for name, spec in PROVIDERS.items() if os.environ.get(spec["env"])]


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str: ...


@dataclass
class RuleBackend:
    """Deterministic, dependency-free reasoning for tests and offline runs."""
    name: str = "rules"

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        role = system.lower()
        if "critic" in role:
            return json.dumps({
                "issues": ["needs an explicit test", "evidence missing for the main claim"],
                "severity": "medium",
            })
        if "security" in role:
            return json.dumps({
                "findings": [], "risk": "low",
                "note": "no dangerous patterns detected by rule backend",
            })
        if "verifier" in role:
            return json.dumps({"checks": [{"name": "runs", "passed": True}]})
        if "judge" in role:
            return json.dumps({"verdict": "revise", "rationale": "evidence requested"})
        return json.dumps({"plan": ["clarify goal", "produce artifact", "verify"]})


class RemoteBackend:
    """OpenAI-compatible chat completions over stdlib urllib (no SDK needed)."""

    def __init__(self, provider: str, model: Optional[str] = None, timeout: int = 60) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        spec = PROVIDERS[provider]
        self.name = provider
        key = os.environ.get(spec["env"])
        if not key:
            raise RuntimeError(f"{spec['env']} is not set; provider {provider} unavailable")
        self._key = key
        self._base = os.environ.get(f"{provider.upper()}_BASE_URL", spec["base"]).rstrip("/")
        self._model = model or spec["model"]
        self._timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return json.dumps({"error": f"{self.name} backend failed: {exc}"})


class LocalBackend:
    """Runs the in-repo ForgeLM as the reasoning engine."""

    name = "local"

    def __init__(self, model, tokenizer, max_new_tokens: int = 120, temperature: float = 0.8) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        import torch

        prompt = f"{system}\n\n{user}\n\n"
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        out = self.model.generate(
            ids,
            max_new_tokens=min(max_tokens, self.max_new_tokens),
            temperature=self.temperature,
            top_k=40,
        )
        text = self.tokenizer.decode(out[0].tolist())
        return text[len(prompt):] if len(text) > len(prompt) else text


def build_backend(spec: str):
    """Resolve a backend spec: 'rules', 'local', or a provider name."""
    if spec == "rules":
        return RuleBackend()
    if spec == "local":
        return None  # caller wires the local model itself
    if spec in PROVIDERS:
        return RemoteBackend(spec)
    raise ValueError(f"unsupported backend spec: {spec!r}")