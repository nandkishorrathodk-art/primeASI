"""The five domain sub-agents plus orchestrator and judge.

Roles are deliberately adversarial rather than cooperative.  Five agents that
agree with each other produce nothing; the value comes from forced disagreement
with a judge that must rule on evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from forge.agents.backends import Backend, RuleBackend
from forge.agents.blackboard import Blackboard
from forge.agents.protocol import Evidence, Kind, Message, Ruling, Verdict

DOMAINS = ["security", "coding", "hacking", "vision", "architecture"]

# Patterns that must never be produced, in any domain, by any agent.
FORBIDDEN_PATTERNS = [
    (r"\b(exploit|payload|shellcode)\s+for\s+(https?://|target)", "targeted exploit against a live host"),
    (r"\b(scan|attack|ddos|bruteforce|brute-force)\s+(a\s+)?(public|live|production|random)\s+(ip|host|site|server)", "offensive action against a third party"),
    (r"\b(reverse\s+shell|botnet|ransomware|keylogger|credential\s+stealer)\b", "malware construction"),
    (r"\b(exfiltrate|steal)\s+(credentials|tokens|passwords|data)\s+from\b", "credential theft"),
]

# Benign, defensive security work.  This is the lane the security/hacking
# agents operate in.
ALLOWED_SECURITY_WORK = [
    "threat modelling",
    "vulnerability analysis of code you own or are authorised to test",
    "CTF challenges and lab environments",
    "hardening, detection rules, fuzzing, static analysis",
    "secure coding review and patch suggestions",
]


class PolicyViolation(Exception):
    pass


def check_policy(text: str) -> Optional[str]:
    """Return a violation reason if the text crosses into harmful territory."""
    low = text.lower()
    for pattern, reason in FORBIDDEN_PATTERNS:
        if re.search(pattern, low):
            return reason
    return None


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction; models sometimes wrap output in prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class Agent:
    name: str
    domain: str
    system_prompt: str
    backend: Backend = field(default_factory=RuleBackend)
    temperature: float = 0.2

    def propose(self, task: str, blackboard: Blackboard) -> Message:
        context = self._context(task, blackboard)
        raw = self.backend.complete(self.system_prompt, context)
        violation = check_policy(raw)
        if violation:
            raise PolicyViolation(f"{self.name} blocked: {violation}")
        return Message(
            kind=Kind.PROPOSAL,
            sender=self.name,
            recipient="orchestrator",
            content=raw,
            confidence=0.6,
            evidence=[Evidence("reasoning", self.name, "agent proposal")],
        )

    def critique(self, target: Message, blackboard: Blackboard) -> Message:
        prompt = (
            f"Review this proposal from {target.sender} for correctness and safety.\n\n"
            f"{target.content}\n\n"
            "Return JSON with 'issues' (list), 'severity' (low|medium|high)."
        )
        raw = self.backend.complete(self.system_prompt, prompt)
        violation = check_policy(raw)
        if violation:
            raise PolicyViolation(f"{self.name} blocked: {violation}")
        data = _extract_json(raw)
        issues = data.get("issues", [])
        severity = str(data.get("severity", "low")).lower()
        conf = {"low": 0.4, "medium": 0.6, "high": 0.85}.get(severity, 0.4)
        return Message(
            kind=Kind.CRITIQUE,
            sender=self.name,
            recipient=target.sender,
            content=json.dumps({"issues": issues, "severity": severity}),
            confidence=conf,
            evidence=[Evidence("review", self.name, f"reviewed {target.id}")],
            refs=[target.id],
        )

    def _context(self, task: str, blackboard: Blackboard) -> str:
        keys = ", ".join(blackboard.keys()) or "(empty)"
        return (
            f"Task: {task}\n"
            f"Known artefacts on blackboard: {keys}\n"
            f"Answer as the {self.domain} specialist. Be concrete."
        )


def build_team(backends: Optional[dict[str, Backend]] = None) -> dict[str, Agent]:
    """Construct the five sub-agents.

    ``backends`` maps domain -> backend, letting each sub-agent reason with a
    different model when the user has the corresponding credentials.
    """
    backends = backends or {}
    specs = {
        "security": (
            "You are the SECURITY agent. You audit designs for weaknesses and "
            "propose hardening. You work only on systems the user owns or is "
            "authorised to test. You never provide targeted exploits, malware, "
            "or instructions to attack third parties. Reply with JSON: "
            '{"assessment": str, "weaknesses": [str], "hardening": [str]}'
        ),
        "coding": (
            "You are the CODING agent. You write correct, tested code and "
            "refuse to claim something works without a test or trace. Reply "
            'with JSON: {"plan": [str], "code": str, "tests": [str]}'
        ),
        "hacking": (
            "You are the OFFENSIVE-SECURITY agent, restricted to authorised "
            "assessments, CTFs and lab targets. You think like an attacker to "
            "improve defences. You refuse real-world targets and malware. "
            'Reply with JSON: {"attack_surface": [str], "detections": [str]}'
        ),
        "vision": (
            "You are the VISION agent. You reason about image encoders, patch "
            "embeddings and multimodal grounding. Reply with JSON: "
            '{"design": str, "risks": [str]}'
        ),
        "architecture": (
            "You are the ARCHITECTURE agent. You reason about MoE routing, "
            "capacity, load balancing, and training efficiency. Reply with "
            'JSON: {"design": str, "tradeoffs": [str]}'
        ),
    }
    return {
        domain: Agent(
            name=f"{domain}-agent",
            domain=domain,
            system_prompt=prompt,
            backend=backends.get(domain, RuleBackend()),
        )
        for domain, prompt in specs.items()
    }


class Judge:
    """Rules on a round.  Cannot accept work that lacks evidence."""

    def __init__(self, backend: Optional[Backend] = None) -> None:
        self.backend = backend or RuleBackend()
        self.system_prompt = (
            "You are the JUDGE. Weigh the proposals and critiques. Accept only "
            "claims backed by evidence. Reply with JSON: "
            '{"verdict": "accept"|"revise"|"reject", "rationale": str, '
            '"blocking_issues": [str], "confidence": float}'
        )

    def rule(self, task: str, messages: list[Message], blackboard: Blackboard) -> Ruling:
        transcript = "\n".join(m.short() for m in messages[-20:])
        raw = self.backend.complete(
            self.system_prompt, f"Task: {task}\n\nTranscript:\n{transcript}"
        )
        data = _extract_json(raw)
        verdict_raw = str(data.get("verdict", "revise")).lower()
        try:
            verdict = Verdict(verdict_raw)
        except ValueError:
            verdict = Verdict.REVISE

        blocking = list(data.get("blocking_issues", []))
        # Hard rule, independent of any model: no evidence, no acceptance.
        unsupported = [m for m in messages if m.kind == Kind.PROPOSAL and not m.evidence]
        if unsupported and verdict is Verdict.ACCEPT:
            verdict = Verdict.REVISE
            blocking.append("proposals accepted without evidence")

        return Ruling(
            verdict=verdict,
            rationale=str(data.get("rationale", "no rationale provided")),
            blocking_issues=blocking,
            confidence=float(data.get("confidence", 0.5) or 0.5),
        )