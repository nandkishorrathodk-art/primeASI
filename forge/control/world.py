"""Bounded perception of the machine.

"Handle the whole computer" requires the controller to observe the machine.
Observing a whole filesystem into a small context is impossible, so the
problem is a *compression* problem, and it needs to be solved explicitly
rather than by hoping a model summarises well on its own.

The digest is hierarchical and budgeted: it walks to a depth limit, caps the
number of entries per directory, ranks by what actually matters (a changed
file beats an unchanged one; a config file beats a cache file), and reports
its own truncation.  A controller that does not know what it could not see is
worse than one that sees less.

Nothing here reads outside the sandbox roots, and nothing reads file
*contents* -- only metadata.  Content enters the controller's context solely
through an explicit, capability-checked ``read``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", ".forge_quarantine", ".ruff_cache", "dist", "build",
}
IGNORED_EXT = {".pyc", ".pyo", ".so", ".o", ".a", ".lock"}

IMPORTANT_EXT = {".py": 3, ".md": 2, ".toml": 3, ".cfg": 3, ".json": 2,
                 ".yaml": 3, ".yml": 3, ".sh": 3, ".txt": 1}

# Extension sniffing only.  This is a heuristic ordering hint, never a claim
# that a file *is* creds-bearing -- treating a filename as proof of content
# would produce false positives that get ignored, which is worse than none.
SENSITIVE_HINTS = (".env", "id_rsa", "id_ed25519", ".pem", ".key",
                   "credentials", "secrets", ".netrc", ".npmrc", ".pypirc")


@dataclass
class Node:
    path: str
    name: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0
    children: list["Node"] = field(default_factory=list)
    child_count: int = 0
    truncated: bool = False
    sensitive_hint: bool = False
    error: str = ""

    def looks_sensitive(self) -> bool:
        low = self.name.lower()
        return any(h in low for h in SENSITIVE_HINTS)


class WorldDigest:
    """A budgeted, ranked view of the sandbox."""

    def __init__(
        self,
        roots: list[str],
        max_depth: int = 3,
        max_children: int = 25,
        max_nodes: int = 400,
    ) -> None:
        self.roots = roots
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_nodes = max_nodes
        self.nodes_seen = 0
        self.truncated_at: list[str] = []

    # ------------------------------------------------------------------
    def build(self) -> list[Node]:
        self.nodes_seen = 0
        self.truncated_at = []
        return [self._walk(r, 0) for r in self.roots if os.path.exists(r)]

    def _walk(self, path: str, depth: int) -> Node:
        name = os.path.basename(path) or path
        try:
            st = os.stat(path)
            is_dir = os.path.isdir(path)
        except OSError as exc:
            return Node(path, name, False, error=str(exc))

        node = Node(path=path, name=name, is_dir=is_dir,
                    size=st.st_size, mtime=st.st_mtime)
        node.sensitive_hint = node.looks_sensitive()
        self.nodes_seen += 1

        if not is_dir or depth >= self.max_depth or self.nodes_seen >= self.max_nodes:
            if is_dir and depth >= self.max_depth:
                node.truncated = True
                self.truncated_at.append(path)
            return node

        try:
            entries = [e for e in os.scandir(path) if not self._skip(e)]
        except OSError as exc:
            node.error = str(exc)
            return node

        node.child_count = len(entries)
        ranked = sorted(entries, key=self._rank, reverse=True)
        for entry in ranked[:self.max_children]:
            if self.nodes_seen >= self.max_nodes:
                node.truncated = True
                self.truncated_at.append(path)
                break
            node.children.append(self._walk(entry.path, depth + 1))

        if len(ranked) > self.max_children:
            node.truncated = True
            self.truncated_at.append(path)
        return node

    def _skip(self, entry: os.DirEntry) -> bool:
        if entry.is_dir(follow_symlinks=False):
            return entry.name in IGNORED_DIRS
        ext = os.path.splitext(entry.name)[1].lower()
        return ext in IGNORED_EXT

    def _rank(self, entry: os.DirEntry) -> tuple:
        """Rank by relevance to a controller, not by name."""
        try:
            st = entry.stat()
            age = time.time() - st.st_mtime
        except OSError:
            return (0, 0, 0, entry.name)
        is_dir = entry.is_dir(follow_symlinks=False)
        ext = os.path.splitext(entry.name)[1].lower()
        return (
            1 if is_dir else 0,                     # structure before detail
            IMPORTANT_EXT.get(ext, 0),              # source/docs before data
            1 if age < 3600 else 0,                 # recently changed files
            entry.name,
        )

    # ------------------------------------------------------------------
    def summary(self, node: Optional[Node] = None, depth: int = 0) -> str:
        """Indented text digest, the form a small LM context can hold."""
        lines: list[str] = []
        for n in (node.children if node else self.build()):
            self._render(n, depth, lines)
        return "\n".join(lines)

    def _render(self, n: Node, depth: int, lines: list[str]) -> None:
        pad = "  " * depth
        if n.error:
            lines.append(f"{pad}{n.name}/ [error: {n.error}]")
        elif n.is_dir:
            more = ""
            if n.child_count > len(n.children):
                more = f" (+{n.child_count - len(n.children)} more, omitted)"
            flag = " [name suggests secrets]" if n.sensitive_hint else ""
            lines.append(f"{pad}{n.name}/{more}{flag}")
            for c in n.children:
                self._render(c, depth + 1, lines)
        else:
            flag = " [name suggests secrets]" if n.sensitive_hint else ""
            lines.append(f"{pad}{n.name}  {n.size}B{flag}")

    def stats(self) -> dict:
        return {
            "nodes": self.nodes_seen,
            "truncated_dirs": self.truncated_at,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
        }