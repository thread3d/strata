#!/usr/bin/env python3
"""Chart the CPU-pool worker sweep: results-threads.json -> a CSV and one SVG.

    python3 bench/macpro51/charts_threads.py

Pure standard library (the sweep machine has no matplotlib; an SVG is crisp and readable as text anyway).
The error bars are +/- 1 sample standard deviation across the runs of that worker count, so the spread of
each point is visible rather than hidden behind a mean; min and max stay in the CSV and the tooltip.
"""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results-threads.json"
OUT = REPO / "docs" / "benchmarks" / "macpro51"

INK, MUTED, GRID, BAR, ERR = "#1f2933", "#6b7280", "#e5e7eb", "#2563eb", "#111827"


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main() -> int:
    recs = [r for r in json.loads(RESULTS.read_text()) if "prefill_tok_s" in r]
    by = {}
    for r in recs:
        by.setdefault(r["workers"], []).append(r)

    rows = []
    for w in sorted(by):
        rs = by[w]
        tok = [r["prefill_tok_s"] for r in rs]
        ms = [r["prefill_ms"] for r in rs]
        dec = [r["decode_ms"] for r in rs if "decode_ms" in r]
        rows.append({
            "workers": w, "runs": len(rs),
            "mean_tok_s": sum(tok) / len(tok), "sd_tok_s": stdev(tok),
            "min_tok_s": min(tok), "max_tok_s": max(tok),
            "mean_ms": sum(ms) / len(ms), "sd_ms": stdev(ms),
            "mean_decode_ms": (sum(dec) / len(dec)) if dec else None,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "thread-sweep.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["workers", "runs", "mean_prefill_tok_s", "sd", "min", "max",
                       "mean_prefill_ms", "sd_ms", "mean_decode_ms"])
        for r in rows:
            wtr.writerow(["%d" % r["workers"], r["runs"], "%.3f" % r["mean_tok_s"], "%.3f" % r["sd_tok_s"],
                          "%.3f" % r["min_tok_s"], "%.3f" % r["max_tok_s"], "%.1f" % r["mean_ms"], "%.1f" % r["sd_ms"],
                          ("%.1f" % r["mean_decode_ms"]) if r["mean_decode_ms"] is not None else ""])

    # ---- SVG
    W, H = 900, 470
    L, R, T, B = 78, 28, 54, 78
    pw, ph = W - L - R, H - T - B
    lo = min(r["mean_tok_s"] - r["sd_tok_s"] for r in rows)
    hi = max(r["mean_tok_s"] + r["sd_tok_s"] for r in rows)
    # near-flat data is the interesting case here: a zero-based axis would draw 13 identical bars and hide
    # that the spread is inside the noise.  Zoom to the data (with a margin) whenever it occupies < 35% of
    # the zero-based axis, and say so in the subtitle.
    zero_based = hi * 1.18
    span = hi - lo
    if span < 0.35 * zero_based:
        pad = max(span * 0.9, hi * 0.01)
        ymin, ymax = max(0.0, lo - pad), hi + pad
        zoomed = True
    else:
        ymin, ymax = 0.0, zero_based
        zoomed = False
    n = len(rows)
    slot = pw / n
    bw = slot * 0.56

    def x_of(i):
        return L + slot * (i + 0.5)

    def y_of(v):
        return T + ph - (v - ymin) / (ymax - ymin) * ph

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img">' % (W, H, W, H),
             '<title>macpro51 Qwen3.8-Flash-Next prefill rate against the CPU expert-pool worker count</title>',
             '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
             '<text x="%d" y="26" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="16" fill="%s">'
             'Prefill rate against CPU expert-pool workers &#8212; ud-iq1_m, 260-token prompt</text>' % (L - 6, INK),
             '<text x="%d" y="44" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="12" fill="%s">'
             'bars = mean of %d runs, error bar = &#177;1 standard deviation; 12 = physical cores, 24 = logical maximum%s</text>'
             % (L - 6, MUTED, max(r["runs"] for r in rows),
                "; y axis zoomed to the data (flat result)" if zoomed else "")]

    ticks = 5
    for k in range(ticks + 1):
        v = ymin + (ymax - ymin) * k / ticks
        y = y_of(v)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1"/>' % (L, y, L + pw, y, GRID))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" font-family="system-ui,sans-serif" font-size="11" fill="%s">%.2f</text>'
                     % (L - 8, y + 4, MUTED, v))
    parts.append('<text x="18" y="%.1f" transform="rotate(-90 18 %.1f)" text-anchor="middle" font-family="system-ui,sans-serif" '
                 'font-size="12" fill="%s">prefill tokens per second</text>' % (T + ph / 2, T + ph / 2, MUTED))

    for i, r in enumerate(rows):
        x = x_of(i)
        y = y_of(r["mean_tok_s"])
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2">'
                     '<title>%d workers: %.2f tok/s mean of %d (%.2f-%.2f), prefill %.1f ms, decode %.0f ms</title></rect>'
                     % (x - bw / 2, y, bw, T + ph - y, BAR, r["workers"], r["mean_tok_s"], r["runs"],
                        r["min_tok_s"], r["max_tok_s"], r["mean_ms"],
                        r["mean_decode_ms"] if r["mean_decode_ms"] is not None else float("nan")))
        if r["sd_tok_s"] > 0:
            lo, hi = y_of(r["mean_tok_s"] - r["sd_tok_s"]), y_of(r["mean_tok_s"] + r["sd_tok_s"])
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>' % (x, lo, x, hi, ERR))
            for yy in (lo, hi):
                parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
                             % (x - bw * 0.22, yy, x + bw * 0.22, yy, ERR))
        parts.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="%s">%d</text>'
                     % (x, T + ph + 18, MUTED, r["workers"]))

    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.2"/>' % (L, T + ph, L + pw, T + ph, INK))
    parts.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="%s">'
                 '--pool-workers</text>' % (L + pw / 2, H - 40, MUTED))
    parts.append('</svg>')

    svg_path = OUT / "thread-sweep.svg"
    svg_path.write_text("\n".join(parts) + "\n")
    print(csv_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
