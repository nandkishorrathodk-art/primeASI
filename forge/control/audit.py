"""Tamper-evident audit chain.

Every action appends a record whose hash covers the previous record's hash.
Modifying or removing any historical entry breaks the chain from that point
onward, which ``verify()`` reports with the exact index where the break
starts.  This is the property that makes the log trustworthy after the fact,
without needing to trust whoever holds the file.

Note the honest limit: this detects tampering, it does not prevent it.  An
attacker who can rewrite the whole file can rebuild a consistent chain.  That
is why the chain head is also printed on every run and can be anchored
externally.  Detection plus external anchoring is the achievable guarantee.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

GENESIS = "0" * 64


@dataclass
class Record:
    index: int
    ts: float
    actor: str
    action: str
    detail: dict
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "index": self.index,
                "ts": round(self.ts, 6),
                "actor": self.actor,
                "action": self.action,
                "detail": self.detail,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def as_dict(self) -> dict:
        return {
            "index": self.index, "ts": self.ts, "actor": self.actor,
            "action": self.action, "detail": self.detail,
            "prev_hash": self.prev_hash, "hash": self.hash,
        }


@dataclass
class VerifyResult:
    ok: bool
    n_records: int
    first_bad_index: Optional[int] = None
    reason: str = ""
    head: str = GENESIS


class AuditChain:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: list[Record] = []
        if path and os.path.exists(path):
            self._load()

    # -- writing --------------------------------------------------------
    def append(self, actor: str, action: str, detail: Optional[dict] = None) -> Record:
        prev = self._records[-1].hash if self._records else GENESIS
        rec = Record(
            index=len(self._records),
            ts=time.time(),
            actor=actor,
            action=action,
            detail=detail or {},
            prev_hash=prev,
        )
        rec.hash = rec.compute_hash()
        self._records.append(rec)
        if self.path:
            self._flush()
        return rec

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(r.as_dict(), default=str) + "\n")

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._records.append(Record(**json.loads(line)))

    # -- verification ----------------------------------------------------
    def verify(self) -> VerifyResult:
        prev = GENESIS
        for i, rec in enumerate(self._records):
            if rec.index != i:
                return VerifyResult(False, len(self._records), i,
                                    "index gap or reordering", self.head())
            if rec.prev_hash != prev:
                return VerifyResult(False, len(self._records), i,
                                    "broken chain link", self.head())
            if rec.compute_hash() != rec.hash:
                return VerifyResult(False, len(self._records), i,
                                    "record contents modified", self.head())
            prev = rec.hash
        return VerifyResult(True, len(self._records), None, "chain intact", self.head())

    def head(self) -> str:
        return self._records[-1].hash if self._records else GENESIS

    def records(self) -> list[Record]:
        return list(self._records)

    def find(self, action: str) -> list[Record]:
        return [r for r in self._records if r.action == action]

    def filter(self, action: str) -> list[Record]:
        return self.find(action)

    def n(self) -> int:
        return len(self._records)

    # -- tamper simulation, used by tests --------------------------------
    def tamper(self, index: int, new_action: str) -> None:
        """Rewrite an entry *without* fixing the chain.  Detection must catch it."""
        self._records[index].action = new_action