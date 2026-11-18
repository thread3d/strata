#!/usr/bin/env python3
"""Qwen3.8-Flash-Next quantisation benchmark sweep on macpro51 (Metal 2, RX 580).

One engine process per run: `strata` opens the GGUF in place, streams the routed experts from the SSD through
its arena, reads a fixed prompt one token at a time and then decodes a fixed number of greedy tokens.  Every
round contains every runnable pack once, in a fresh shuffled order, so a drift in the machine is spread over the
matrix instead of being assigned to one pack.  After every completed run the results, the page and its SVG
charts are rewritten, committed and pushed.

    python3 bench/macpro51/run_bench_qwen38.py --rounds 5 --seed 20261118
    python3 bench/macpro51/run_bench_qwen38.py --only Q2_0 --rounds 1 --no-push
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import charts  # noqa: E402

RESULTS = HERE / "results-qwen38.jsonl"
DOC = REPO / "docs" / "macpro51-qwen38-benchmarks.md"
CHART_DIR = REPO / "docs" / "benchmarks" / "macpro51"

HOME = Path.home()
BIN = HOME / "strata-bench" / "build-bench" / "strata"
MODELS = HOME / "models" / "qwen38"
PACK_TOOL = REPO / "tools" / "iq_pack.py"
PY = "python3"

PREFILL, DECODE = 32, 32
PREFILL_ONLY = False   # --prefill-only: one token after the prompt, so the prompt is measured and no decode rate is claimed
# Qwen3.6's ids (the same 248 320-entry Qwen vocabulary): the first 32 of
#   "Explain, in two short sentences, why the sky looks blue during the day and red at sunset. Then say
#    why the Moon looks orange near the horizon."
IDS = [3, 1, 42498, 456, 11, 303, 1330, 2716, 22157, 11, 3069, 279, 12515, 5686, 6105, 2261, 279, 1834, 321,
       2438, 506, 41564, 13, 4844, 1910, 3069, 279, 16820, 5686, 18016, 3043, 279]

# (folder under ~/models/qwen38, label, quantisation, source, note)
CONFIGS = [
    ("Q2_0", "Q2_0", "Q2_0 (GSQ-RCO)", "ISTA-DASLab", "2-bit ternary experts, the smallest ISTA pack"),
    ("IQ2_XS", "IQ2_XS", "IQ2_XS (GSQ-RCO)", "ISTA-DASLab", "2-bit i-quant experts"),
    ("IQ3_XXS", "IQ3_XXS", "IQ3_XXS (GSQ-RCO)", "ISTA-DASLab", "3-bit i-quant experts"),
    ("IQ3_S", "IQ3_S", "IQ3_S (GSQ-RCO)", "ISTA-DASLab", "3.5-bit i-quant experts, the quality reference here"),
    ("UD-IQ1_M", "UD-IQ1_M", "UD-IQ1_M (Unsloth v3.1)", "unsloth", "~1-bit dynamic"),
    ("UD-Q2_0", "UD-Q2_0", "UD-Q2_0 (Unsloth v3.1)", "unsloth", "2-bit dynamic"),
    ("UD-IQ2_XS", "UD-IQ2_XS", "UD-IQ2_XS (Unsloth v3.1)", "unsloth", "2-bit i-quant dynamic"),
    ("UD-Q2_K_XL", "UD-Q2_K_XL", "UD-Q2_K_XL (Unsloth v3.1)", "unsloth", "2-bit dynamic, some 3-bit experts"),
    ("UD-IQ4_XS", "UD-IQ4_XS", "UD-IQ4_XS (Unsloth)", "unsloth", "~4-bit i-quant experts"),
    ("UD-Q4_K_XL", "UD-Q4_K_XL", "UD-Q4_K_XL (Unsloth)", "unsloth", "4-bit K-quant experts"),
    ("UD-Q5_K_XL", "UD-Q5_K_XL", "UD-Q5_K_XL (Unsloth)", "unsloth", "5-bit K-quant experts"),
    ("UD-Q6_K_XL", "UD-Q6_K_XL", "UD-Q6_K_XL (Unsloth)", "unsloth", "6-bit K-quant experts"),
    ("Q8_0", "Q8_0", "Q8_0 (Unsloth)", "unsloth", "8-bit, the reference quality"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, nd=2):
    if 0 < abs(x) < 1.0:
        nd = max(nd, 3)
    return f"{x:.{nd}f}"


def pack_dir(label: str) -> Path:
    return MODELS / label


def shards(label: str):
    d = pack_dir(label)
    if not d.is_dir():
        return None, None
    one = sorted(d.glob("*-00001-of-*.gguf")) or sorted(d.glob("*.gguf"))
    if not one:
        return None, None
    two = sorted(d.glob("*-00002-of-*.gguf"))
    return one[0], (two[0] if two else one[0])


def pack_size(label: str) -> int:
    d = pack_dir(label)
    if not d.is_dir():
        return 0
    return sum(p.stat().st_size for p in d.glob("*.gguf"))


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=check)


def git_head() -> str:
    try:
        return git("rev-parse", "--short=10", "HEAD").stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------------------------------------ preparation

def prepare_pack(label: str) -> str:
    """Build the metadata pack (native_experts.txt, index.txt, dense.bin) beside the GGUF, once."""
    d = pack_dir(label)
    one, _ = shards(label)
    if one is None:
        return "not downloaded"
    pack = d / "pack"
    if (pack / "native_experts.txt").exists() and (pack / "index.txt").exists():
        return "ready"
    cmd = [PY, str(PACK_TOOL), "--gguf", str(one), "--out", str(pack)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        return "pack failed: " + ((proc.stderr.strip() or proc.stdout.strip()).splitlines() or [""])[-1]
    return "ready"


# ------------------------------------------------------------------------------------------------------- runs

def run_one(cfg, round_no, order):
    label = cfg[1]
    t0 = time.time()
    rec = {"label": label, "quant": cfg[2], "family": "qwen38", "kind": "q38",
           "source": cfg[3], "note": cfg[4], "round": round_no, "order": order,
           "prefill_tokens": PREFILL, "decode_tokens": DECODE}
    one, two = shards(label)
    if one is None:
        rec.update(rc=-1, error="not downloaded")
        return rec
    prepared = prepare_pack(label)
    rec["pack_state"] = prepared
    if prepared != "ready":
        rec.update(rc=-2, error=prepared)
        return rec
    ids = ",".join(str(i) for i in IDS[:PREFILL])
    decode_n = 1 if PREFILL_ONLY else DECODE
    # One token per verify window, no drafter: every pack runs the same policy, and the window path is the
    # quantisation's own cost.  The prompt goes through the engine's prompt path in one chunk of PREFILL.
    cmd = [str(BIN), "--pack", str(pack_dir(label) / "pack"), "--native", str(one),
           "--ple-gguf", str(two), "--tokens", ids, "--max-new", str(decode_n),
           "--max-context", "512", "--spec", "1", "--suffix-draft", "0", "--lookup-max-t", "0",
           "--prefill", str(PREFILL)]
    env = dict(os.environ)
    env.setdefault("STRATA_METAL_STREAM", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5400, env=env)
    out, err = proc.stdout, proc.stderr
    rec["rc"] = proc.returncode
    dm = re.search(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", out, re.M)
    pm = re.search(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", out, re.M)
    om = re.search(r"^output\s*:\s*(.*)$", out, re.M)
    lm = re.search(r"dense tensors read into memory \(([\d.]+) GB, ([\d.]+) s", err)
    if lm:
        rec["dense_gb"], rec["dense_load_s"] = float(lm.group(1)), float(lm.group(2))
    if proc.returncode != 0 or dm is None:
        rec.update(error=(err.strip().splitlines() or [""])[-1] or "no decode line",
                   stderr_tail=err[-2000:], stdout_tail=out[-800:])
        return rec
    rec["decode_tokens"] = int(dm.group(1))
    rec["decode_s"] = float(dm.group(2)) / 1000.0
    rec["decode_ok"] = (not PREFILL_ONLY) and int(dm.group(1)) > 1
    rec["decode_tok_s"] = int(dm.group(1)) / rec["decode_s"] if rec["decode_s"] > 0 else 0.0
    if pm:
        rec["prefill_tokens"] = int(pm.group(1))
        rec["prefill_s"] = float(pm.group(2)) / 1000.0
        rec["prefill_tok_s"] = int(pm.group(1)) / rec["prefill_s"] if rec["prefill_s"] > 0 else 0.0
    if om:
        toks = [int(x) for x in om.group(1).split() if re.fullmatch(r"-?\d+", x)]
        rec["generated"] = toks[:DECODE]
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------------------------------------------- records

def load_records():
    if not RESULTS.exists():
        return []
    out = []
    for ln in RESULTS.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def rate(r, which):
    if r.get(which + "_s"):
        n = r.get(which + "_tokens", 0)
        if which == "decode" and n > 1:
            return n / r[which + "_s"]
        if which == "prefill":
            return n / r[which + "_s"]
    return r.get(which + "_tok_s", 0.0)


def aggregate(records):
    by = {}
    for r in records:
        by.setdefault(r["label"], []).append(r)
    agg = {}
    for label, rs in by.items():
        ok = [r for r in rs if r.get("rc") == 0 and r.get("prefill_s")]
        dec = [r for r in ok if r.get("decode_ok")]
        steady_p = [r for r in ok if r.get("round", 0) > 0]
        steady_d = [r for r in dec if r.get("round", 0) > 0]
        use_p = steady_p if len(steady_p) >= 2 else ok
        use_d = steady_d if len(steady_d) >= 2 else dec
        agg[label] = {"runs": sorted(rs, key=lambda r: r.get("order", 0)),
                      "decode": [rate(r, "decode") for r in use_d],
                      "prefill": [rate(r, "prefill") for r in use_p],
                      "decode_all": [rate(r, "decode") for r in dec],
                      "prefill_all": [rate(r, "prefill") for r in ok],
                      "n": len(ok), "n_decode": len(dec),
                      "n_steady": len(steady_p),
                      "used_steady": len(steady_p) >= 2,
                      "failed": len(rs) - len(ok)}
    return agg


# ------------------------------------------------------------------------------------------------------ charts

def prompt_only(agg) -> bool:
    """True while no pack has a usable decode measurement (the forward is not answering yet)."""
    return not any(a.get("n_decode") for a in agg.values())


def render_charts(agg):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for cfg in CONFIGS:
        a = agg.get(cfg[1])
        if not a or not a["n"]:
            continue
        items.append({"label": cfg[1], "mean": mean(a["prefill"]), "std": stdev(a["prefill"]),
                      "points": a["prefill"], "n": a["n"]})
    if items:
        charts.grouped_bar_chart(
            CHART_DIR / "qwen38-prefill.svg",
            "Qwen3.8-Flash-Next prompt throughput on macpro51",
            "The fixed prompt read through the streaming path; steady state (rounds 2+), mean ± 1 sample standard deviation; dots are individual runs",
            [("qwen38", items)], "tokens per second (prompt)", "higher is better")
    if prompt_only(agg):
        pts = [{"family": "qwen38", "label": c[1], "x": pack_size(c[0]) / 1073741824.0,
                "y": mean(agg[c[1]]["prefill"]), "std": stdev(agg[c[1]]["prefill"])}
               for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n"]]
        if pts:
            charts.scatter_chart(
                CHART_DIR / "qwen38-size-vs-prompt.svg",
                "Qwen3.8-Flash-Next: prompt throughput against pack size",
                "What a token reads from the SSD and the format's decode cost matter more than the file size alone",
                pts, "pack size on disk (GiB)", "prompt tokens/s")
        return
    items = [{"label": c[1], "mean": mean(agg[c[1]]["decode"]), "std": stdev(agg[c[1]]["decode"]),
              "points": agg[c[1]]["decode"], "n": agg[c[1]]["n_decode"]}
             for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n_decode"]]
    if items:
        charts.grouped_bar_chart(
            CHART_DIR / "qwen38-decode.svg",
            "Qwen3.8-Flash-Next decode throughput on macpro51",
            "RX 580 8 GB, Metal 2, experts streamed from the SSD; greedy; steady state (rounds 2+), mean ± 1 sample standard deviation; dots are individual runs",
            [("qwen38", items)], "tokens per second (decode)", "higher is better")
    pts = [{"family": "qwen38", "label": c[1], "x": pack_size(c[0]) / 1073741824.0,
            "y": mean(agg[c[1]]["decode"]), "std": stdev(agg[c[1]]["decode"])}
           for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n_decode"]]
    if pts:
        charts.scatter_chart(
            CHART_DIR / "qwen38-size-vs-decode.svg",
            "Qwen3.8-Flash-Next: decode throughput against pack size",
            "What a token reads from the SSD and the format's decode cost matter more than the file size alone",
            pts, "pack size on disk (GiB)", "decode tokens/s")


def render_timeline(records, agg):
    which = "prefill" if prompt_only(agg) else "decode"
    rows = [{"order": r.get("order", 0), "label": r["label"], "family": r["family"], "value": rate(r, which)}
            for r in sorted(records, key=lambda r: r.get("order", 0)) if r.get("rc") == 0 and r.get("prefill_s")]
    if rows:
        charts.timeline_chart(
            CHART_DIR / "qwen38-run-order.svg",
            "Every Qwen3.8 run in the order it was made (%s)" % which,
            "Each round reshuffles the packs, so a drift in the machine is shared rather than assigned to one pack",
            rows)


# --------------------------------------------------------------------------------------------------------- doc

def render_doc_prompt_only(agg, records, meta):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append("# Qwen3.8-Flash-Next prompt-throughput benchmarks on macpro51 (RX 580, Metal 2)")
    L.append("")
    L.append(f"_Generated {now} by `bench/macpro51/run_bench_qwen38.py` after every completed run. Every number below was produced on the machine named here._")
    L.append("")
    L.append("> **The engine does not answer yet.** Its forward pass goes non-finite at the first full-attention layer, so")
    L.append("> there is no decode rate to report and none is claimed here. What this page measures is the prompt path: the")
    L.append("> whole model is opened, the dense side and the expert arena are built, and the 32-token prompt is read through")
    L.append("> the same streaming kernels a decode window uses. Those bytes and that work are real. The blocker and the")
    L.append("> evidence are in [qwen38-metal-macpro51.md](qwen38-metal-macpro51.md); this page becomes the decode matrix")
    L.append("> the day the query projection is fixed.")
    L.append("")
    L.append("## The machine")
    L.append("")
    L.append("| | |")
    L.append("| --- | --- |")
    L.append("| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |")
    L.append("| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |")
    L.append("| OS | macOS 11.7.11 (Big Sur, 20G1443) |")
    L.append("| Engine | `strata` 0.1.59-metal, branch `macpro51` |")
    L.append(f"| Branch / commit at the last run | `macpro51` / `{meta.get('git_head', 'unknown')}` |")
    L.append("")
    L.append("## What ran")
    L.append("")
    L.append("| Pack | Quantisation | Source | On disk | Runs | Prompt, steady (mean ± sd) | All runs (mean) | Steady range |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cfg in CONFIGS:
        a = agg.get(cfg[1])
        size = pack_size(cfg[0])
        if not a or not a["n"]:
            L.append(f"| `{cfg[1]}` | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | — | — | — | — |")
            continue
        p = a["prefill"]
        star = "" if a.get("used_steady") else "*"
        L.append(f"| `{cfg[1]}`{star} | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | {a['n']} | "
                 f"{fmt(mean(p))} ± {fmt(stdev(p))} | {fmt(mean(a['prefill_all']))} | {fmt(min(p))} – {fmt(max(p))} |")
    L.append("")
    L.append("\\* fewer than two steady-state runs (rounds 2+) so far, so this row uses every completed run.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- **One process per run.** The engine reads the dense side (2.36 GB for Q2_0), fills the expert arena (2.2 GB),")
    L.append(f"  compiles its window kernels, then reads the {PREFILL}-token prompt as one chunk through the prompt path.")
    L.append(f"- **The same {PREFILL} token ids** for every pack (the Qwen 248 320-entry vocabulary, a varied English sentence).")
    L.append(f"- **Mixed order.** Each round contains every downloaded pack once, in a fresh shuffled order (seed in")
    L.append(f"  `bench/macpro51/meta-qwen38.json`), so a drift in the machine is spread rather than assigned to one pack.")
    L.append(f"- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs (rounds 2+). Round 1 is each")
    L.append(f"  pack's first pass on a cold page cache and is kept separately. The dots on the chart are the individual runs.")
    L.append(f"- **Streaming.** `STRATA_METAL_STREAM=1`: the pack cannot be resident in 8 GB of VRAM, and every pack is measured")
    L.append(f"  on the same streamed path. The prompt path and the decode window read the same experts from the same arena.")
    L.append("")
    L.append("## Prompt throughput")
    L.append("")
    L.append("![Prompt throughput by quantisation](benchmarks/macpro51/qwen38-prefill.svg)")
    L.append("")
    L.append("## Size against prompt speed")
    L.append("")
    L.append("![Prompt throughput against pack size](benchmarks/macpro51/qwen38-size-vs-prompt.svg)")
    L.append("")
    L.append("A prompt token reads the same 10 routed experts per layer a decode token does, so the pack's encoding and the")
    L.append("bytes a token actually touches decide this number more than the file size does.")
    L.append("")
    L.append("## What the numbers say")
    L.append("")
    done = [(c[1], agg[c[1]]) for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n"]]
    if len(done) >= 2:
        pre = sorted(((l, mean(a["prefill"])) for l, a in done), key=lambda x: -x[1])
        L.append(f"- **Fastest prompt read:** `{pre[0][0]}` at {pre[0][1]:.2f} tok/s; **slowest:** `{pre[-1][0]}` at {pre[-1][1]:.2f} tok/s ({pre[0][1]/max(1e-9, pre[-1][1]):.2f}× between them).")
        cvs = sorted(((l, 100.0 * stdev(a["prefill"]) / mean(a["prefill"])) for l, a in done if a["n"] > 1 and mean(a["prefill"]) > 0), key=lambda x: -x[1])
        if cvs:
            L.append(f"- **Run-to-run spread:** coefficient of variation from {cvs[-1][1]:.1f}% (`{cvs[-1][0]}`) to {cvs[0][1]:.1f}% (`{cvs[0][0]}`).")
        L.append("")
        L.append("The first round is the first pass over each pack (cold page cache, cold clocks); the later rounds are the")
        L.append("steady state. The round-1 and steady numbers are kept apart in the table and the per-run log.")
    else:
        L.append("_Not enough completed packs yet; this section fills in as the rounds finish._")
    L.append("")
    L.append("## Run order (mixing)")
    L.append("")
    L.append("![Every run in execution order](benchmarks/macpro51/qwen38-run-order.svg)")
    L.append("")
    L.append("The sweep is resumable: a `(round, pack)` pair already in `results-qwen38.jsonl` is skipped, so an interrupted")
    L.append("or rebooted machine costs one run rather than the series.")
    L.append("")
    L.append("## Per-run log")
    L.append("")
    L.append("| # | Round | Pack | Prompt tokens | Prompt s | Prompt tok/s | Note |")
    L.append("| ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for r in sorted(records, key=lambda r: r.get("order", 0)):
        if r.get("rc") == 0 and r.get("prefill_s"):
            note = "first round (cold)" if r.get("round", 0) == 0 else "later round"
            L.append(f"| {r.get('order', 0)+1} | {r.get('round', 0)+1} | `{r['label']}` | {r.get('prefill_tokens', 0)} | "
                     f"{fmt(r['prefill_s'])} | {fmt(rate(r, 'prefill'))} | {note} |")
        else:
            L.append(f"| {r.get('order', 0)+1} | {r.get('round', 0)+1} | `{r.get('label','?')}` | — | — | — | failed: {r.get('error', 'unknown')} |")
    L.append("")
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")


def render_doc(agg, records, meta):
    if prompt_only(agg):
        render_doc_prompt_only(agg, records, meta)
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append("# Qwen3.8-Flash-Next quantisation benchmarks on macpro51 (RX 580, Metal 2)")
    L.append("")
    L.append(f"_Generated {now} by `bench/macpro51/run_bench_qwen38.py` after every completed run. Every number below was produced on the machine named here._")
    L.append("")
    L.append("## The machine")
    L.append("")
    L.append("| | |")
    L.append("| --- | --- |")
    L.append("| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |")
    L.append("| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |")
    L.append("| OS | macOS 11.7.11 (Big Sur, 20G1443) |")
    L.append("| Engine | `strata` 0.1.59-metal (`metal/` build, Ninja Release), branch `macpro51` |")
    L.append(f"| Branch / commit at the last run | `macpro51` / `{meta.get('git_head', 'unknown')}` |")
    L.append("")
    L.append("The engine's Metal decode window reads the model's GGUF in place. The RX 580 cannot hold a 40–160 GB pack in 8 GB of VRAM, so the dense side (norms, hyper-connection projections, the shared expert and the attention/GDN projections) is read once into host memory and the routed experts stream from the SSD through an arena; a token's experts are looked up after its router and the next layer's are read ahead. Nothing is copied into a second file.")
    L.append("")
    L.append("## What ran")
    L.append("")
    L.append("Qwen3.8-Flash-Next (`qwen4exp`) is the shape the engine compiles for: 48 trunk layers (36 gated-DeltaNet, 12 full-attention QSA with the 4×128 indexer), 4 hyper-connection streams of rank 320, and 512 routed experts of width 640 with 10 used per token, plus the 28.8 GB PLE n-gram table in the second shard. Every pack below is the same architecture; only the expert encodings differ.")
    L.append("")
    L.append("| Pack | Quantisation | Source | On disk | Result |")
    L.append("| --- | --- | --- | ---: | --- |")
    for cfg in CONFIGS:
        label = cfg[1]
        a = agg.get(label)
        size = pack_size(cfg[0])
        if a and a["n"]:
            verdict = f"{a['n']} run{'s' if a['n'] != 1 else ''}, {mean(a['decode']):.2f} tok/s decode (steady state)"
        elif not size:
            verdict = "not downloaded"
        else:
            verdict = "queued"
        L.append(f"| `{label}` | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | {verdict} |")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- **One process per run.** `strata generate` opens the shard, builds the pack metadata once, reads the dense side and the expert arena, then times the prompt and the decode separately. Load time is reported on its own and is not part of either throughput.")
    L.append(f"- **Fixed workload.** The same {PREFILL} token ids for every pack (the Qwen 248 320-entry vocabulary, a varied English sentence rather than repeated ids, so routing is not narrowed), read one at a time, then {DECODE} greedy tokens.")
    L.append(f"- **Mixed order.** Each round contains every downloaded pack once, in a fresh shuffled order (seed in the run's `meta.json`). No pack is always first or last, so thermal drift, page-cache state and SSD behaviour are spread across the matrix.")
    L.append(f"- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs (rounds 2+); round 1 is each pack's first pass and is priced separately. Individual runs are the dots on the charts.")
    L.append(f"- **Streaming.** `STRATA_METAL_STREAM=1` is set for every run: the pack cannot be resident in 8 GB of VRAM, and forcing the streamed path keeps every pack on the same code path.")
    L.append("")
    L.append("## Decode throughput")
    L.append("")
    L.append("![Decode throughput by quantisation](benchmarks/macpro51/qwen38-decode.svg)")
    L.append("")
    L.append("| Pack | Steady runs | Decode, steady (mean ± sd) | All runs (mean) | Steady range | Prefill, steady (mean ± sd) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cfg in CONFIGS:
        a = agg.get(cfg[1])
        if not a or not a["n"]:
            continue
        d, p = a["decode"], a["prefill"]
        L.append(f"| `{cfg[1]}` | {a['n_steady'] if a['used_steady'] else a['n']} | {fmt(mean(d))} ± {fmt(stdev(d))} | "
                 f"{fmt(mean(a['decode_all']))} | {fmt(min(d))} – {fmt(max(d))} | {fmt(mean(p))} ± {fmt(stdev(p))} |")
    L.append("")
    L.append("## Prompt (prefill) throughput")
    L.append("")
    L.append("![Prefill throughput by quantisation](benchmarks/macpro51/qwen38-prefill.svg)")
    L.append("")
    L.append("The prompt is read one token at a time on this port, so prefill here is per-token latency through the same expert stream, not the batched GEMM a CUDA card runs. It is comparable across these packs, not to a desktop GPU's numbers.")
    L.append("")
    L.append("## Size against speed")
    L.append("")
    L.append("![Decode throughput against pack size](benchmarks/macpro51/qwen38-size-vs-decode.svg)")
    L.append("")
    L.append("A token reads only its 10 routed experts per layer, so a bigger file is not automatically slower: what matters is the bytes one token actually touches, the encoding's decode cost (Q2_0 and the i-quants trade a wider row read for cheap decode), and how much of the dense side stays resident.")
    L.append("")
    L.append("## What the numbers say")
    L.append("")
    done = [(c[1], agg[c[1]]) for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n"]]
    if len(done) >= 2:
        dec = sorted(((l, mean(a["decode"])) for l, a in done), key=lambda x: -x[1])
        L.append(f"- **Fastest decode:** `{dec[0][0]}` at {dec[0][1]:.2f} tok/s; **slowest:** `{dec[-1][0]}` at {dec[-1][1]:.2f} tok/s ({dec[0][1]/max(1e-9, dec[-1][1]):.2f}× between them).")
        cvs = sorted(((l, 100.0 * stdev(a["decode"]) / mean(a["decode"])) for l, a in done if a["n"] > 1 and mean(a["decode"]) > 0), key=lambda x: -x[1])
        if cvs:
            L.append(f"- **Run-to-run spread:** coefficient of variation from {cvs[-1][1]:.1f}% (`{cvs[-1][0]}`) to {cvs[0][1]:.1f}% (`{cvs[0][0]}`).")
        pre = sorted(((l, mean(a["prefill"])) for l, a in done), key=lambda x: -x[1])
        L.append(f"- **Fastest prompt read:** `{pre[0][0]}` at {pre[0][1]:.2f} tok/s; slowest `{pre[-1][0]}` at {pre[-1][1]:.2f} tok/s.")
        rows = []
        for cfg in CONFIGS:
            a = agg.get(cfg[1])
            if not a or not a["n"]:
                continue
            r1 = [rate(r, "decode") for r in a["runs"] if r.get("round") == 0 and r.get("rc") == 0]
            rn = [rate(r, "decode") for r in a["runs"] if r.get("round", 0) > 0 and r.get("rc") == 0]
            if r1 and rn:
                rows.append((cfg[1], mean(r1), mean(rn)))
        if rows:
            ch = [100.0 * (b - a) / max(1e-9, a) for _, a, b in rows]
            L.append("")
            L.append(f"The first round is the first pass over each pack (cold page cache, cold GPU clocks); the later rounds are the steady state. The difference is the price of a first run, not a property of the pack: it averages {mean(ch):+.1f}% over these {len(rows)} pack{'s' if len(rows) != 1 else ''}.")
            L.append("")
            L.append("| Pack | Round 1 (tok/s) | Rounds 2+ (tok/s) | Change |")
            L.append("| --- | ---: | ---: | ---: |")
            for l, a, b in rows:
                L.append(f"| `{l}` | {a:.2f} | {b:.2f} | {100.0*(b-a)/max(1e-9,a):+.1f}% |")
    else:
        L.append("_Not enough completed packs yet; this section fills in as the rounds finish._")
    L.append("")
    L.append("## Run order (mixing)")
    L.append("")
    L.append("![Every run in execution order](benchmarks/macpro51/qwen38-run-order.svg)")
    L.append("")
    L.append("The sweep is resumable: a `(round, pack)` pair already in `results-qwen38.jsonl` is skipped, so an interrupted or rebooted machine costs one run rather than the series.")
    L.append("")
    L.append("## Per-run log")
    L.append("")
    L.append("| # | Round | Pack | Prefill s | Decode s | Decode tok/s | Note |")
    L.append("| ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for r in sorted(records, key=lambda r: r.get("order", 0)):
        if r.get("rc") == 0 and "decode_s" in r:
            note = "first round (cold)" if r.get("round", 0) == 0 else "later round"
            L.append(f"| {r.get('order', 0)+1} | {r.get('round', 0)+1} | `{r['label']}` | {fmt(r.get('prefill_s', 0))} | {fmt(r['decode_s'])} | {fmt(rate(r, 'decode'))} | {note} |")
        else:
            L.append(f"| {r.get('order', 0)+1} | {r.get('round', 0)+1} | `{r.get('label','?')}` | — | — | — | failed: {r.get('error', 'unknown')} |")
    L.append("")
    L.append("## Limits recorded here")
    L.append("")
    L.append("- The pack's second shard is the 28.8 GB PLE n-gram table. It is read from the SSD a few rows at a time and is never loaded whole; its size counts in the download, not in VRAM.")
    L.append("- Metal 2 on this card has no device addresses; kernels take buffer + offset, and the engine hands every weight that way. No Metal 3 feature is used.")
    L.append("- The output head is 512 wide in the draft path and 248 320 wide in the sample; greedy decoding is used throughout so the comparison is not a sampler's.")
    L.append("")
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prefill-only", action="store_true",
                    help="measure the prompt only (one token after it); for a forward that does not answer yet")
    a = ap.parse_args()
    global PREFILL_ONLY
    PREFILL_ONLY = a.prefill_only

    rng = random.Random(a.seed)
    listed = [c for c in CONFIGS if a.only is None or c[1] in a.only]
    if a.list:
        for c in listed:
            print(c[1], "downloaded" if shards(c[1])[0] else "missing")
        return 0
    # a pack that is not on disk yet is not a run: it would only add a failed row to the log
    active = [c for c in listed if shards(c[1])[0] is not None]
    if not active:
        print("no pack on disk: nothing to run")
        return 0

    records = load_records()
    done = {(r.get("round"), r.get("label")) for r in records if r.get("rc") == 0}
    meta = {"host": "macpro51", "rounds": a.rounds, "seed": a.seed, "prefill": PREFILL, "decode": DECODE,
            "git_head": git_head(), "prefill_only": PREFILL_ONLY,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (HERE / "meta-qwen38.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    order = 0
    for rnd in range(a.rounds):
        batch = list(active)
        rng.shuffle(batch)
        for cfg in batch:
            label = cfg[1]
            if (rnd, label) in done:
                continue
            print(f"[{datetime.now().strftime('%H:%M:%S')}] round {rnd+1}/{a.rounds}  order {order}  {label}", flush=True)
            rec = run_one(cfg, rnd, order)
            rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rec["git_head"] = meta["git_head"]
            with RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            records.append(rec)
            order += 1
            print(f"    rc={rec.get('rc')} decode={fmt(rate(rec, 'decode'))} tok/s "
                  f"prefill={fmt(rate(rec, 'prefill'))} tok/s {rec.get('error','')}", flush=True)
            agg = aggregate(records)
            render_charts(agg)
            render_timeline(records, agg)
            render_doc(agg, records, meta)
            if not a.no_push:
                git("add", "bench/macpro51/results-qwen38.jsonl", "bench/macpro51/meta-qwen38.json",
                    "bench/macpro51/run_bench_qwen38.py", "docs/macpro51-qwen38-benchmarks.md",
                    "docs/benchmarks/macpro51/")
                msg = (f"[Benchmark] macpro51 qwen38: {label} run {rnd+1}/{a.rounds} "
                       f"({fmt(rate(rec, 'decode'))} tok/s decode)")
                git("commit", "-m", msg)
                git("push", "origin", "macpro51")
                print("    committed and pushed", flush=True)
    agg = aggregate(records)
    render_charts(agg)
    render_timeline(records, agg)
    render_doc(agg, records, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
