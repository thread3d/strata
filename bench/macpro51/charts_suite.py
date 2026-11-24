#!/usr/bin/env python3
"""Chart the suite: prompt-size scaling (with error bars) and the threads x prompt matrix.

    python3 bench/macpro51/charts_suite.py

Writes, under docs/benchmarks/macpro51/:
    suite-scaling.csv / suite-scaling.svg   prefill rate and wall time against prompt size (log2 x axis)
    suite-threads.csv                        the --pool-workers x prompt-size matrix
The edge cases are a table in the doc, not a chart: their point is the verdict, not a number.
"""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results-suite.json"
OUT = REPO / "docs" / "benchmarks" / "macpro51"
INK, MUTED, GRID, LINE, ERR = "#1f2933", "#6b7280", "#e5e7eb", "#2563eb", "#111827"


def esc(v):
    return html.escape(str(v), quote=True)


def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def scaling_svg(points, path):
    """points: [(tokens, mean_tok_s, sd_tok_s, mean_wall, n)] sorted by tokens."""
    W, H = 900, 480
    L, R, T, B = 82, 96, 58, 76
    pw, ph = W - L - R, H - T - B
    xs = [p[0] for p in points]
    x0, x1 = math.log2(xs[0]), math.log2(xs[-1])
    ymax = max(p[1] + p[2] for p in points) * 1.15

    def X(t):
        return L + (math.log2(t) - x0) / max(1e-9, (x1 - x0)) * pw

    def Y(v):
        return T + ph - v / ymax * ph

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img">' % (W, H, W, H),
           '<title>macpro51 UD-IQ1_M prefill rate against prompt size</title>',
           '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
           '<text x="%d" y="26" font-family="system-ui,sans-serif" font-size="16" fill="%s">Prefill rate against prompt size &#8212; '
           'ud-iq1_m, dflash2 attached</text>' % (L - 8, INK),
           '<text x="%d" y="44" font-family="system-ui,sans-serif" font-size="12" fill="%s">'
           'points = mean of up to 2 runs, error bar = &#177;1 s.d. (smaller than the marker at every point); right labels = wall time per run</text>' % (L - 8, MUTED)]

    for k in range(6):
        v = ymax * k / 5
        y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>' % (L, y, L + pw, y, GRID))
        out.append('<text x="%d" y="%.1f" text-anchor="end" font-family="system-ui,sans-serif" font-size="11" fill="%s">%.1f</text>'
                   % (L - 8, y + 4, MUTED, v))
    out.append('<text x="20" y="%.1f" transform="rotate(-90 20 %.1f)" text-anchor="middle" font-family="system-ui,sans-serif" '
               'font-size="12" fill="%s">prefill tokens per second</text>' % (T + ph / 2, T + ph / 2, MUTED))

    pts = " ".join("%.1f,%.1f" % (X(t), Y(m)) for t, m, _, _, _ in points)
    out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (LINE, pts))
    wall_labels = []
    for t, m, sd, wall, n in points:
        x, y = X(t), Y(m)
        if sd > 0:
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>' % (x, Y(m - sd), x, Y(m + sd), ERR))
            for yy in (Y(m - sd), Y(m + sd)):
                out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>' % (x - 4, yy, x + 4, yy, ERR))
        out.append('<circle cx="%.1f" cy="%.1f" r="3.6" fill="%s"><title>%d tokens: %.2f tok/s mean of %d, wall %.1f s</title></circle>'
                   % (x, y, LINE, t, m, n, wall))
        out.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="%s">%d</text>'
                   % (x, T + ph + 18, MUTED, t))
        wall_labels.append([y + 4, "%.0f s" % wall])
    # space the wall-time labels 13 px apart so a steep curve cannot overlap them
    wall_labels.sort()
    for k in range(1, len(wall_labels)):
        if wall_labels[k][0] - wall_labels[k - 1][0] < 13:
            wall_labels[k][0] = wall_labels[k - 1][0] + 13
    for ly, txt in wall_labels:
        out.append('<text x="%d" y="%.1f" font-family="system-ui,sans-serif" font-size="10" fill="%s">%s</text>'
                   % (L + pw + 8, min(ly, T + ph), MUTED, txt))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.2"/>' % (L, T + ph, L + pw, T + ph, INK))
    out.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="%s">prompt tokens (log scale)</text>'
               % (L + pw / 2, H - 34, MUTED))
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n")


def main():
    recs = [r for r in json.loads(RESULTS.read_text()) if r["verdict"] == "ran"]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- scaling
    by = {}
    for r in recs:
        if r["block"] == "scale" and "prefill_tok_s" in r:
            by.setdefault(r["prompt_tokens"], []).append(r)
    points = []
    rows = []
    for t in sorted(by):
        rs = by[t]
        tok = [x["prefill_tok_s"] for x in rs]
        wall = [x["wall_s"] for x in rs]
        ms = [x["prefill_ms"] for x in rs]
        points.append((t, sum(tok) / len(tok), stdev(tok), sum(wall) / len(wall), len(rs)))
        rows.append([t, len(rs), "%.3f" % (sum(tok) / len(tok)), "%.3f" % stdev(tok),
                     "%.0f" % (sum(ms) / len(ms)), "%.0f" % stdev(ms), "%.1f" % (sum(wall) / len(wall))])
    with open(OUT / "suite-scaling.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_tokens", "runs", "mean_prefill_tok_s", "sd", "mean_prefill_ms", "sd_ms", "mean_wall_s"])
        w.writerows(rows)
    if points:
        scaling_svg(points, OUT / "suite-scaling.svg")

    # ---- threads x prompt
    with open(OUT / "suite-threads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_workers", "prompt_tokens", "runs", "mean_prefill_tok_s", "mean_prefill_ms", "mean_wall_s"])
        grid = {}
        for r in recs:
            if r["block"] == "threads" and "prefill_tok_s" in r:
                grid.setdefault((r["workers"], r["prompt_tokens"]), []).append(r)
        for (wk, t) in sorted(grid):
            rs = grid[(wk, t)]
            w.writerow([wk, t, len(rs), "%.3f" % (sum(x["prefill_tok_s"] for x in rs) / len(rs)),
                        "%.0f" % (sum(x["prefill_ms"] for x in rs) / len(rs)),
                        "%.1f" % (sum(x["wall_s"] for x in rs) / len(rs))])

    print(OUT / "suite-scaling.csv")
    print(OUT / "suite-scaling.svg")
    print(OUT / "suite-threads.csv")


if __name__ == "__main__":
    main()
