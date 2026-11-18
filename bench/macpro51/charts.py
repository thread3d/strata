#!/usr/bin/env python3
"""Vector charts for the macpro51 quantisation sweep.

Pure standard library: the machine that runs the sweep has no matplotlib, and an SVG is the better
deliverable anyway (crisp at any zoom, readable in the repository, and no per-run image build).
"""

from __future__ import annotations

import html
import math
from pathlib import Path

FONT = "Helvetica, Arial, sans-serif"
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
AXIS = "#9ca3af"

# One colour per family; the packs inside a family are ordered light-to-heavy in the caller.
FAMILY_COLOURS = {
    "qwen36": "#2563eb",
    "dsv41": "#c2410c",
    "qwen38": "#0f766e",
}


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if v <= m * mag:
            return m * mag
    return 10.0 * mag


class Svg:
    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w, self.h = width, height
        self.parts: list[str] = []
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">'
        )
        self.parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
        self.parts.append(
            f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-family="{FONT}" '
            f'font-size="21" font-weight="600" fill="{INK}">{_esc(title)}</text>'
        )
        if subtitle:
            self.parts.append(
                f'<text x="{width/2:.1f}" y="58" text-anchor="middle" font-family="{FONT}" '
                f'font-size="13" fill="{MUTED}">{_esc(subtitle)}</text>'
            )

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, size: int = 12, anchor: str = "start",
             fill: str = INK, weight: str = "400", rotate: float | None = None) -> None:
        t = (f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else "")
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}"{t}>{_esc(s)}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = AXIS,
             width: float = 1.0, dash: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, opacity: float = 1.0) -> None:
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, w):.1f}" height="{max(0.0, h):.1f}" '
                 f'fill="{fill}" opacity="{opacity:.2f}"/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "white",
               width: float = 0.8, opacity: float = 1.0) -> None:
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}" opacity="{opacity:.2f}"/>')

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def grouped_bar_chart(path: Path, title: str, subtitle: str, groups: list[tuple[str, list[dict]]],
                      ylabel: str, note: str = "") -> None:
    """groups: [(family, [{label, mean, std, points, n}, ...]), ...]."""
    left, right, top, bottom = 92, 34, 92, 150
    svg = Svg(1180, 620, title, subtitle)
    plot_w = svg.w - left - right
    plot_h = svg.h - top - bottom

    flat = [item for _, items in groups for item in items]
    if not flat:
        svg.save(path)
        return
    hi = max(item["mean"] + item["std"] for item in flat)
    ymax = _nice_max(hi * 1.12)

    # grid + y labels
    ticks = 6
    for i in range(ticks + 1):
        v = ymax * i / ticks
        y = top + plot_h - plot_h * i / ticks
        svg.line(left, y, left + plot_w, y, GRID if i else AXIS)
        svg.text(left - 10, y + 4, f"{v:.2f}" if ymax < 10 else f"{v:.1f}", 12, "end", MUTED)
    svg.text(22, top + plot_h / 2, ylabel, 13, "middle", INK, "600", rotate=-90)

    # slots: a gap between families is 0.8 of a slot
    slots: list[tuple[str, dict | None]] = []
    for gi, (family, items) in enumerate(groups):
        if gi:
            slots.append(("gap", None))
            slots.append(("gap", None))
        for it in items:
            slots.append((family, it))
    slot_w = plot_w / max(1, len(slots))
    bar_w = min(64.0, slot_w * 0.58)
    base = top + plot_h

    for si, (family, item) in enumerate(slots):
        if item is None:
            continue
        cx = left + slot_w * (si + 0.5)
        colour = FAMILY_COLOURS.get(family, "#334155")
        m, sd = item["mean"], item["std"]
        h = plot_h * m / ymax
        svg.rect(cx - bar_w / 2, base - h, bar_w, h, colour, 0.86)
        # error bar: mean +/- one sample standard deviation
        if sd > 0:
            y0, y1 = base - plot_h * (m + sd) / ymax, base - plot_h * max(0.0, m - sd) / ymax
            svg.line(cx, y0, cx, y1, INK, 1.4)
            svg.line(cx - 11, y0, cx + 11, y0, INK, 1.4)
            svg.line(cx - 11, y1, cx + 11, y1, INK, 1.4)
        # individual runs as jittered points, so the spread the std describes is visible
        pts = item.get("points") or []
        for pi, p in enumerate(pts):
            jitter = ((pi % 5) - 2) * (bar_w * 0.11) if len(pts) > 1 else 0.0
            y = base - plot_h * p / ymax
            svg.circle(cx + jitter, y, 2.6, "white", colour, 1.0, 0.95)
        svg.text(cx, base - h - 9, f"{m:.2f}", 11, "middle", INK, "600")
        svg.text(cx, base + 20, item["label"], 11, "end", INK, "500", rotate=-28)

    svg.line(left, base, left + plot_w, base, AXIS, 1.4)
    # legend
    lx = left
    for family, _ in groups:
        svg.rect(lx, 76, 12, 12, FAMILY_COLOURS.get(family, "#334155"), 0.86)
        svg.text(lx + 18, 86, family, 12, "start", INK, "500")
        lx += 34 + 7.0 * len(family)
    if note:
        svg.text(svg.w - right, svg.h - 12, note, 11, "end", MUTED)
    svg.save(path)


def scatter_chart(path: Path, title: str, subtitle: str, points: list[dict], xlabel: str, ylabel: str) -> None:
    """points: [{family, label, x, y, std, points}]."""
    left, right, top, bottom = 92, 32, 92, 78
    svg = Svg(1040, 540, title, subtitle)
    plot_w, plot_h = svg.w - left - right, svg.h - top - bottom
    if not points:
        svg.save(path)
        return
    xmax = _nice_max(max(p["x"] for p in points) * 1.08)
    ymax = _nice_max(max(p["y"] + p["std"] for p in points) * 1.12)
    for i in range(7):
        v = ymax * i / 6
        y = top + plot_h - plot_h * i / 6
        svg.line(left, y, left + plot_w, y, GRID if i else AXIS)
        svg.text(left - 10, y + 4, f"{v:.2f}", 12, "end", MUTED)
    for i in range(7):
        v = xmax * i / 6
        x = left + plot_w * i / 6
        svg.line(x, top, x, top + plot_h, GRID if i else AXIS)
        svg.text(x, top + plot_h + 20, f"{v:.0f}", 12, "middle", MUTED)
    svg.text(24, top + plot_h / 2, ylabel, 13, "middle", INK, "600", rotate=-90)
    svg.text(left + plot_w / 2, svg.h - 26, xlabel, 13, "middle", INK, "600")

    for p in points:
        colour = FAMILY_COLOURS.get(p["family"], "#334155")
        x = left + plot_w * p["x"] / xmax
        y = top + plot_h - plot_h * p["y"] / ymax
        if p["std"] > 0:
            y0 = top + plot_h - plot_h * (p["y"] + p["std"]) / ymax
            y1 = top + plot_h - plot_h * max(0.0, p["y"] - p["std"]) / ymax
            svg.line(x, y0, x, y1, colour, 1.2)
            svg.line(x - 8, y0, x + 8, y0, colour, 1.2)
            svg.line(x - 8, y1, x + 8, y1, colour, 1.2)
        svg.circle(x, y, 5.0, colour, "white", 1.2)
        svg.text(x + 9, y - 7, p["label"], 10, "start", INK, "500")
    svg.save(path)


def timeline_chart(path: Path, title: str, subtitle: str, rows: list[dict]) -> None:
    """rows: [{order, label, family, value}] in execution order."""
    left, right, top, bottom = 92, 190, 92, 82
    svg = Svg(1180, 560, title, subtitle)
    plot_w, plot_h = svg.w - left - right, svg.h - top - bottom
    if not rows:
        svg.save(path)
        return
    ymax = _nice_max(max(r["value"] for r in rows) * 1.12)
    for i in range(7):
        v = ymax * i / 6
        y = top + plot_h - plot_h * i / 6
        svg.line(left, y, left + plot_w, y, GRID if i else AXIS)
        svg.text(left - 10, y + 4, f"{v:.2f}", 12, "end", MUTED)
    svg.text(24, top + plot_h / 2, "decode tokens/s", 13, "middle", INK, "600", rotate=-90)
    n = max(1, len(rows))
    for i, r in enumerate(rows):
        x = left + plot_w * (i + 0.5) / n
        y = top + plot_h - plot_h * r["value"] / ymax
        colour = FAMILY_COLOURS.get(r["family"], "#334155")
        svg.circle(x, y, 3.4, colour, "white", 0.9)
        if i % 5 == 0 or i == n - 1:
            svg.text(x, top + plot_h + 18, str(r["order"] + 1), 9, "middle", MUTED)
    # right-hand legend: one entry per distinct label, in first-seen order
    seen: list[tuple[str, str]] = []
    for r in rows:
        if (r["family"], r["label"]) not in seen:
            seen.append((r["family"], r["label"]))
    for i, (family, label) in enumerate(seen):
        y = top + 10 + i * 19
        svg.circle(left + plot_w + 22, y, 4.2, FAMILY_COLOURS.get(family, "#334155"), "white", 0.9)
        svg.text(left + plot_w + 34, y + 4, label, 11, "start", INK)
    svg.text(left + plot_w / 2, svg.h - 22, "run number (execution order, rounds shuffled)",
             12, "middle", MUTED)
    svg.save(path)
