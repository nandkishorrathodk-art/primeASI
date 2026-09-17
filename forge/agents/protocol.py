"""Inter-agent message protocol."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Kind(str, Enum):
    TASK = "task"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    SECURITY_FINDING = "security_finding"
    VERIFICATION = "verification"
    REVISION = "revision"
    RULING = "ruling"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


@dataclass
class Evidence:
    """A verifiable reference.  Claims without evidence are not admissible."""
    kind: str                      # "test", "trace", "doc", "static_analysis"
    ref: str                       # e.g. "pytest::test_roundtrip" or a URL
    detail: str = ""


@dataclass
class Message:
    kind: Kind
    sender: str
    recipient: str
    content: str
    confidence: float = 0.5
    evidence: list[Evidence] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)   # message ids this replies to
    verdict: Optional[Verdict] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def short(self) -> str:
        return f"[{self.sender}->{self.recipient} {self.kind.value}] {self.content[:80]}"


@dataclass
class Ruling:
    """Final judge output for a round."""
    verdict: Verdict
    rationale: str
    blocking_issues: list[str] = field(default_factory=list)
    confidence: float = 0.5