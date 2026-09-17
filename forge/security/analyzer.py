"""Sandboxed static analysis.  Never executes submitted code.

Rationale: an agentic coding loop that *runs* model-generated code is a
remote-code-execution hole by design.  This module reads and pattern-matches
instead, so the analysis path has no execution capability at all.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Finding:
    rule: str
    severity: str          # low | medium | high
    line: int
    detail: str


DANGEROUS_CALLS = {
    "eval": "high", "exec": "high", "compile": "medium",
    "__import__": "high", "os.system": "high", "os.popen": "high",
    "subprocess.call": "medium", "subprocess.run": "medium",
    "subprocess.Popen": "medium", "pickle.loads": "high",
    "marshal.loads": "high", "input": "low",
}

PATTERN_RULES: list[tuple[str, str, str]] = [
    (r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{3,}['\"]",
     "HARDCODED_CREDENTIAL", "high"),
    (r"(?i)(execute|exec)\s*\(\s*['\"].*(select|insert|update|delete).*['\"]",
     "SQL_INJECTION_RISK", "high"),
    (r"(?i)(md5|sha1)\s*\(", "WEAK_HASH", "medium"),
    (r"(?i)random\.(random|randint|choice)\s*\(", "INSECURE_RANDOM", "medium"),
    (r"(?i)verify\s*=\s*False", "TLS_VERIFY_DISABLED", "high"),
    (r"(?i)shell\s*=\s*True", "SHELL_INJECTION_RISK", "high"),
    (r"(?i)yaml\.load\s*\((?![^)]*Loader)", "UNSAFE_YAML_LOAD", "high"),
    (r"(?i)assert\s+.*(password|auth|permission)", "ASSERT_FOR_SECURITY", "medium"),
]


def scan_source(source: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()

    for i, line in enumerate(lines, start=1):
        for pattern, rule, severity in PATTERN_RULES:
            if re.search(pattern, line):
                findings.append(Finding(rule, severity, i, line.strip()[:120]))

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        findings.append(Finding("SYNTAX_ERROR", "medium", exc.lineno or 0, str(exc)))
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DANGEROUS_CALLS:
                findings.append(Finding(
                    f"DANGEROUS_CALL:{name}", DANGEROUS_CALLS[name],
                    getattr(node, "lineno", 0), name,
                ))
    return findings


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def risk_score(findings: list[Finding]) -> float:
    weights = {"low": 0.1, "medium": 0.35, "high": 0.8}
    if not findings:
        return 0.0
    raw = sum(weights.get(f.severity, 0.1) for f in findings)
    return min(1.0, raw / (1.0 + 0.5 * len(findings)))