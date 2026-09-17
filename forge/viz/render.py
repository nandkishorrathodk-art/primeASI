"""Text-mode visualisation, no plotting dependencies required."""
from __future__ import annotations

from typing import Iterable, Sequence


def _block(v: float) -> str:
    ramp = " .:-=+*#%@"
    v = max(0.0, min(1.0, v))
    return ramp[int(v * (len(ramp) - 1))]


def render_image(img, width: int = 32) -> str:
    """img: torch tensor (C, H, W) or (H, W) -> ASCII art string."""
    try:
        import torch
    except ImportError:
        return "[torch unavailable]"
    t = img.detach().cpu()
    if t.dim() == 3:
        t = t.mean(0)
    rows = []
    h, w = t.shape
    for y in range(h):
        rows.append("".join(_block(float(t[y, x])) for x in range(0, w, max(1, w // width))))
    return "\n".join(rows)


def sparkline(values: Sequence[float], width: int = 60) -> str:
    if not values:
        return ""
    ramp = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    lo, hi = min(sampled), max(sampled)
    span = (hi - lo) or 1.0
    return "".join(ramp[int((v - lo) / span * (len(ramp) - 1))] for v in sampled)


def bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "█" * filled + "·" * (width - filled)


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    sep = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)), sep]
    out += [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def render_attention(weights: Sequence[float], labels: Sequence[str]) -> str:
    """Horizontal bar chart of routing/attention weights."""
    return "\n".join(
        f"{lab:>16} | {bar(w)} {w:.3f}" for lab, w in zip(labels, weights)
    )