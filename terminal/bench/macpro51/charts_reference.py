#!/usr/bin/env python3
"""Wall clock for the identical 47-step workload: strata on Metal vs llama.cpp on the CPU.

Reads results-qwen36.json and results-reference.json, writes the CSV and the SVG beside them
(docs/benchmarks/macpro51/).  Standard library only, like charts.py.

    python3 terminal/bench/macpro51/charts_reference.py
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next((q for q in HERE.parents if (q / ".git").exists()), HERE.parents[1])   # the clone root, wherever bench/ lives
OUT = REPO / "docs" / "benchmarks" / "macpro51"
STRATA = HERE / "results-qwen36.json"
REFERENCE = HERE / "results-reference.json"

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
STRATA_COL = "#2563eb"
LLAMA_COL = "#c2410c"


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def rows():
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else []
    strata = json.loads(STRATA.read_text()) if STRATA.is_file() else []

    llama = {}
    for r in ref:
        if "wall_s" in r:
            llama[r["pack"]] = min(llama.get(r["pack"], 1e9), r["wall_s"])

    # Run 1 of each pack pays the cold page cache, so the warm runs are the fair comparison.
    walls = {}
    for r in strata:
        if "wall_s" in r and r["round"] > 1:
            walls.setdefault(r["pack"], []).append(r["wall_s"])

    out = []
    for pack, ws in walls.items():
        if pack not in llama:
            continue
        out.append({"pack": pack, "strata": min(ws), "llama": llama[pack]})
    out.sort(key=lambda r: r["strata"])
    return out


def write_csv(data) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "same-workload-wall.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pack", "strata_wall_s", "llamacpp_wall_s", "ratio_strata_over_llamacpp"])
        for r in data:
            w.writerow([r["pack"], "%.1f" % r["strata"], "%.1f" % r["llama"],
                        "%.2f" % (r["strata"] / r["llama"])])
    return path


def write_svg(data) -> Path:
    left, right, top = 232, 120, 62
    row_h, bar_h = 46, 14
    x_max = 60.0
    width = 880
    height = top + row_h * len(data) + 96
    plot = width - left - right

    def sx(v):
        return left + plot * (min(v, x_max) / x_max)

    p = []
    add = p.append
    add('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Helvetica, Arial, sans-serif">' % (width, height, width, height))
    add('<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height))
    add('<text x="%d" y="26" font-size="15" font-weight="600" fill="%s">'
        'Same 47-step workload, wall clock: strata (Metal) vs llama.cpp (CPU)</text>' % (left, INK))
    add('<text x="%d" y="44" font-size="11" fill="%s">23-token context and 24 greedy tokens on the '
        'same pack; model load included on both sides</text>' % (left, MUTED))

    for tick in range(0, 61, 10):
        x = sx(tick)
        add('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>'
            % (x, top - 6, x, top + row_h * len(data), GRID))
        add('<text x="%.1f" y="%d" font-size="10" fill="%s" text-anchor="middle">%d</text>'
            % (x, top + row_h * len(data) + 14, MUTED, tick))

    for i, r in enumerate(data):
        y = top + i * row_h
        add('<text x="16" y="%d" font-size="12" font-weight="600" fill="%s">%s</text>'
            % (y + 12, INK, esc(r["pack"])))
        add('<text x="16" y="%d" font-size="10" fill="%s">strata is %.1fx slower</text>'
            % (y + 26, MUTED, r["strata"] / r["llama"]))
        add('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="%s">'
            '<title>strata: %.1f s</title></rect>'
            % (left, y + 2, sx(r["strata"]) - left, bar_h, STRATA_COL, r["strata"]))
        add('<text x="%.1f" y="%d" font-size="10" fill="%s">%.1f s</text>'
            % (sx(r["strata"]) + 5, y + 13, INK, r["strata"]))
        add('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="%s">'
            '<title>llama.cpp: %.1f s</title></rect>'
            % (left, y + 19, sx(r["llama"]) - left, bar_h, LLAMA_COL, r["llama"]))
        add('<text x="%.1f" y="%d" font-size="10" fill="%s">%.1f s</text>'
            % (sx(r["llama"]) + 5, y + 30, INK, r["llama"]))

    legend_y = top + row_h * len(data) + 34
    add('<rect x="%d" y="%d" width="10" height="10" rx="2" fill="%s"/>' % (left, legend_y - 8, STRATA_COL))
    add('<text x="%d" y="%d" font-size="10" fill="%s">strata-q36-metal (Metal 2)</text>'
        % (left + 15, legend_y, MUTED))
    add('<rect x="%d" y="%d" width="10" height="10" rx="2" fill="%s"/>'
        % (left + 190, legend_y - 8, LLAMA_COL))
    add('<text x="%d" y="%d" font-size="10" fill="%s">llama.cpp 8bed7cfb9 (CPU + BLAS)</text>'
        % (left + 205, legend_y, MUTED))
    add('<text x="16" y="%d" font-size="10" fill="%s">Shorter is better. The llama.cpp build has no '
        'Metal backend on this machine, so it is a CPU reference.</text>' % (legend_y + 22, MUTED))
    add("</svg>")

    path = OUT / "same-workload-wall.svg"
    path.write_text("\n".join(p) + "\n", encoding="utf-8")
    return path


def main() -> int:
    data = rows()
    if not data:
        print("no paired data; run run_matrix.py and run_reference.py first")
        return 1
    print(write_csv(data))
    print(write_svg(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
