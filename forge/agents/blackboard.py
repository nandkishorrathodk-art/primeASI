"""Shared blackboard with provenance tracking.

Every write is a typed entry attributed to an agent and backed by evidence.
This is what makes the multi-agent debate auditable: at any point you can ask
"who claimed this, on what basis, and did anything contradict it?".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from forge.agents.protocol import Evidence, Message, Kind


@dataclass
class Entry:
    key: str
    value: Any
    author: str
    evidence: list[Evidence] = field(default_factory=list)
    supersedes: Optional[str] = None      # key of the entry this replaces
    revision: int = 1
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "key": self.key, "author": self.author, "revision": self.revision,
            "value": self.value, "evidence": [e.ref for e in self.evidence],
            "supersedes": self.supersedes,
        }


class Blackboard:
    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._history: list[Entry] = []
        self._messages: list[Message] = []

    # -- writes ---------------------------------------------------------
    def write(
        self,
        key: str,
        value: Any,
        author: str,
        evidence: Optional[list[Evidence]] = None,
        supersedes: Optional[str] = None,
    ) -> Entry:
        prev = self._entries.get(key)
        rev = (prev.revision + 1) if prev else 1
        entry = Entry(
            key=key, value=value, author=author,
            evidence=evidence or [], supersedes=supersedes, revision=rev,
        )
        self._entries[key] = entry
        self._history.append(entry)
        return entry

    def post(self, msg: Message) -> Message:
        self._messages.append(msg)
        return msg

    # -- reads ----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        e = self._entries.get(key)
        return e.value if e else default

    def entry(self, key: str) -> Optional[Entry]:
        return self._entries.get(key)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def messages(self, kind: Optional[Kind] = None) -> list[Message]:
        if kind is None:
            return list(self._messages)
        return [m for m in self._messages if m.kind == kind]

    def history(self) -> list[Entry]:
        return list(self._history)

    def unsupported_claims(self) -> list[Entry]:
        """Entries written without a single piece of evidence."""
        return [e for e in self._entries.values() if not e.evidence]

    # -- introspection ---------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "entries": {k: v.as_dict() for k, v in self._entries.items()},
            "n_messages": len(self._messages),
            "n_entries": len(self._entries),
            "unsupported": [e.key for e in self.unsupported_claims()],
        }

    def provenance(self, key: str) -> list[dict]:
        """Full revision chain for a key, oldest first."""
        chain = [e.as_dict() for e in self._history if e.key == key]
        return chain