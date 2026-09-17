"""Self-contained HTML report generator (inline SVG, no dependencies)."""
from __future__ import annotations

import html
import json
import os
from typing import Sequence


def _svg_line(values: Sequence[float], w: int = 640, h: int = 180,
              color: str = "#4ade80") -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * (w - 20) + 10
        y = h - 20 - (v - lo) / span * (h - 40)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" '
        f'points="{" ".join(pts)}"/></svg>'
    )


def _svg_bars(values: Sequence[float], labels: Sequence[str],
              w: int = 640, h: int = 180, color: str = "#60a5fa") -> str:
    if not values:
        return ""
    n = len(values)
    bw = (w - 40) / n
    peak = max(values) or 1.0
    bars = []
    for i, (v, lab) in enumerate(zip(values, labels)):
        bh = (v / peak) * (h - 50)
        x = 20 + i * bw
        y = h - 30 - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" '
            f'height="{bh:.1f}" fill="{color}" rx="3"/>'
            f'<text x="{x + bw * 0.35:.1f}" y="{h - 12}" font-size="11" '
            f'fill="#94a3b8" text-anchor="middle">{html.escape(str(lab))}</text>'
            f'<text x="{x + bw * 0.35:.1f}" y="{y - 5:.1f}" font-size="10" '
            f'fill="#cbd5e1" text-anchor="middle">{v:.3f}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" class="chart">{"".join(bars)}</svg>'


CSS = """
:root { color-scheme: dark; }
body { background:#0b1120; color:#e2e8f0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
       margin:0; padding:32px; line-height:1.5; }
h1 { font-size:22px; margin:0 0 4px; color:#f8fafc; }
h2 { font-size:15px; margin:28px 0 10px; color:#93c5fd; text-transform:uppercase;
     letter-spacing:.08em; }
.sub { color:#64748b; font-size:12px; margin-bottom:24px; }
.card { background:#111c33; border:1px solid #1e293b; border-radius:10px;
        padding:16px 18px; margin-bottom:16px; }
.chart { width:100%; height:auto; display:block; }
pre { background:#0b1120; border:1px solid #1e293b; border-radius:8px; padding:12px;
      overflow-x:auto; font-size:12px; color:#cbd5e1; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid #1e293b; }
th { color:#94a3b8; font-weight:500; }
.ok { color:#4ade80; } .warn { color:#fbbf24; } .bad { color:#f87171; }
"""


def build_report(out_path: str, sections: list[tuple[str, str]],
                 meta: dict | None = None) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta = meta or {}
    body = "".join(
        f'<h2>{html.escape(title)}</h2><div class="card">{content}</div>'
        for title, content in sections
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Forge report</title>
<style>{CSS}</style></head><body>
<h1>Forge training report</h1>
<div class="sub">{html.escape(json.dumps(meta))}</div>
{body}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


def loss_section(values: Sequence[float]) -> str:
    return _svg_line(values, color="#4ade80")


def expert_section(values: Sequence[float], labels: Sequence[str]) -> str:
    return _svg_bars(values, labels, color="#60a5fa")