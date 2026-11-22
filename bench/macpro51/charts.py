#!/usr/bin/env python3
"""Draw the macpro51 Qwen3.6 numbers: a CSV and one SVG, both from results-qwen36.json.

Pure standard library on purpose - the sweep machine has no matplotlib, and an SVG is the better
deliverable in a repository anyway: it is crisp at any zoom and readable as text.

    python3 bench/macpro51/charts.py
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results-qwen36.json"
OUT = REPO / "docs" / "benchmarks" / "macpro51"

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BAR = "#2563eb"

# The down-projection type is the one property that tracks the two answers (see the report); for
# the packs that never start there is no slab line, so the type comes from the refusal itself.
REFUSED_DOWN = {"UD-Q2_K_XL": "Q5_K (embed)", "UD-Q8_K_XL": "BF16"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def summarise():
    records = json.loads(RESULTS.read_text())
    packs = {}
    for rec in records:
        packs.setdefault(rec["pack"], []).append(rec)

    rows = []
    for pack, runs in packs.items():
        timed = [r for r in runs if "s_per_step" in r]
        steps = sorted(r["s_per_step"] for r in timed)
        row = {
            "pack": pack,
            "bytes": runs[0].get("bytes", 0),
            "gib": round(runs[0].get("bytes", 0) / 1073741824.0, 1),
            "steps": steps,
        }
        slab = next((r.get("slab") for r in timed if r.get("slab")), "")
        if "down" in slab:
            row["down"] = slab.split(",")[-1].strip().replace(" down", "")
        else:
            row["down"] = REFUSED_DOWN.get(pack, "-")
        if steps:
            row["status"] = "runs"
            row["mean"] = sum(steps) / len(steps)
            row["lo"], row["hi"] = steps[0], steps[-1]
        else:
            row["status"] = "refused"
            row["why"] = (runs[0].get("error") or "").strip()
        rows.append(row)
    rows.sort(key=lambda r: r["bytes"])
    return rows


def write_csv(rows) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "decode-steps.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pack", "down", "bytes", "gib", "runs", "mean_s_per_step", "min", "max", "status"])
        for r in rows:
            w.writerow([r["pack"], r["down"], r["bytes"], r["gib"], len(r["steps"]),
                        "%.3f" % r["mean"] if r["steps"] else "",
                        "%.3f" % r["lo"] if r["steps"] else "",
                        "%.3f" % r["hi"] if r["steps"] else "",
                        r["status"]])
    return path


def write_svg(rows) -> Path:
    timed = [r for r in rows if r["steps"]]
    refused = [r for r in rows if not r["steps"]]
    x_max = 1.30
    left, right, top = 232, 96, 58
    row_h, bar_h = 44, 15
    height = top + row_h * len(timed) + 78 + 20 * len(refused)
    width = 860
    plot = width - left - right

    def sx(v):
        return left + plot * (min(v, x_max) / x_max)

    parts = []
    add = parts.append
    add('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Helvetica, Arial, sans-serif">' % (width, height, width, height))
    add('<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height))
    add('<text x="%d" y="26" font-size="15" font-weight="600" fill="%s">'
        'Qwen3.6-35B-A3B on macpro51 (RX 580, Metal 2): seconds per step</text>' % (left, INK))
    add('<text x="%d" y="44" font-size="11" fill="%s">23-token prompt + 24 greedy tokens, 47 steps; '
        'bar = mean of the runs, whisker = min to max</text>' % (left, MUTED))

    for tick in [0.0, 0.3, 0.6, 0.9, 1.2]:
        x = sx(tick)
        add('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>'
            % (x, top - 6, x, top + row_h * len(timed), GRID))
        add('<text x="%.1f" y="%d" font-size="10" fill="%s" text-anchor="middle">%.1f</text>'
            % (x, top + row_h * len(timed) + 14, MUTED, tick))

    for i, r in enumerate(timed):
        y = top + i * row_h
        add('<text x="16" y="%d" font-size="12" font-weight="600" fill="%s">%s</text>'
            % (y + 13, INK, esc(r["pack"])))
        add('<text x="16" y="%d" font-size="10" fill="%s">down %s \u00b7 %.1f GiB</text>'
            % (y + 27, MUTED, esc(r["down"]), r["gib"]))
        add('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="%s">'
            '<title>%s: %.3f s/step over %d runs (%s)</title></rect>'
            % (left, y + 8, sx(r["mean"]) - left, bar_h, BAR, esc(r["pack"]), r["mean"],
               len(r["steps"]), " / ".join("%.3f" % s for s in r["steps"])))
        add('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.5"/>'
            % (sx(r["lo"]), y + 3, sx(r["hi"]), y + 3, MUTED))
        add('<text x="%.1f" y="%d" font-size="11" fill="%s">%.3f</text>'
            % (sx(r["hi"]) + 6, y + 20, INK, r["mean"]))

    legend_y = top + row_h * len(timed) + 34
    add('<text x="16" y="%d" font-size="10" fill="%s">Whisker is the spread across runs; the '
        'reference comparison is in same-workload-wall.svg</text>' % (legend_y, MUTED))

    note_y = legend_y + 24
    for r in refused:
        add('<text x="16" y="%d" font-size="10" fill="%s">%s refused: %s</text>'
            % (note_y, MUTED, esc(r["pack"]), esc(r["why"])[:96]))
        note_y += 14
    add("</svg>")

    path = OUT / "decode-steps.svg"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def main() -> int:
    rows = summarise()
    print(write_csv(rows))
    print(write_svg(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
