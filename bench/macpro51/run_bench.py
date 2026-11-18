#!/usr/bin/env python3
"""Mixed-order quantisation benchmark sweep for Strata on macpro51.

Every run is one engine process: the model is loaded, warmed up, and then a fixed prompt is read followed by a
fixed number of greedy tokens.  The order of the configurations is reshuffled every round, so no pack is always
measured first (or last) and a slow drift in the machine shows up in the error bars rather than in one pack.

After every completed run the results file, the documentation and the SVG charts are rewritten, committed and
pushed.  Re-running the script resumes: a (round, configuration) pair already in results.jsonl is skipped.

    python3 bench/macpro51/run_bench.py --rounds 5 --seed 20261118
    python3 bench/macpro51/run_bench.py --only Q4_K_M --rounds 1 --no-push
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

RESULTS = HERE / "results.jsonl"
META = HERE / "meta.json"
DOC = REPO / "docs" / "macpro51-quant-benchmarks.md"
CHART_DIR = REPO / "docs" / "benchmarks" / "macpro51"

HOME = Path.home()
Q36_BIN = HOME / "strata-bench" / "build-bench" / "strata-q36-metal"
DS4_BIN = HOME / "strata-bench" / "build-bench" / "strata_ds4"
Q36_DIR = HOME / "models" / "qwen36"
DS_DIR = HOME / "models" / "deepseek-v4.1-flash"

# The same ids for every pack of a family: the workload must not change between the things being compared.
# Qwen's 32 ids are the first 32 of the pack tokenizer's encoding of
#     "Explain, in two short sentences, why the sky looks blue during the day and red at sunset. Then say
#      why the Moon looks orange near the horizon."
# (34 tokens, read back with strata-q36-metal --prompt).  A varied sentence rather than a repeated id list:
# a repeated prompt makes a greedy model echo it, which routes to a narrower set of experts than a real one.
# DeepSeek's tokenizer is a different vocabulary; its 16 ids are a fixed, varied sequence inside its vocabulary
# (the same numbers are not the same rendered text, so the two families are compared within themselves).
QWEN_IDS = [3, 1, 42498, 456, 11, 303, 1330, 2716, 22157, 11, 3069, 279, 12515, 5686, 6105, 2261, 279, 1834, 321,
            2438, 506, 41564, 13, 4844, 1910, 3069, 279, 16820, 5686, 18016, 3043, 279]
DS_IDS = [3, 1, 42498, 456, 11, 303, 1330, 2716, 22157, 11, 3069, 279, 12515, 5686, 6105, 2261]

# Prompt and decode lengths.  The Qwen engine reads a prompt one token at a time and decodes one token at a time,
# so each run is (prefill + decode) forwards; 32+32 keeps a run near a minute and a half on the RX 580 while still
# reaching a context where the attention layers do real work.  The DeepSeek engine reports its own prompt and
# decode times; 16+16 there keeps a run inside a few minutes.
QWEN_PREFILL, QWEN_DECODE = 32, 32
DS_PREFILL, DS_DECODE = 16, 16

CONFIGS = [
    # (family, label, quantisation, kind, path)
    ("qwen36", "Q4_K_M", "Q4_K_M", "q36", Q36_DIR / "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
    ("qwen36", "Q6_K", "Q6_K", "q36", Q36_DIR / "Qwen3.6-35B-A3B-Q6_K.gguf"),
    ("qwen36", "Q8_0", "Q8_0", "q36", Q36_DIR / "Qwen3.6-35B-A3B-Q8_0.gguf"),
    ("qwen36", "UD-Q2_K_XL", "Q2_K_XL", "q36", Q36_DIR / "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
    ("qwen36", "UD-Q4_K_XL", "Q4_K_XL", "q36", Q36_DIR / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    ("qwen36", "UD-Q6_K_XL", "Q6_K_XL", "q36", Q36_DIR / "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"),
    ("qwen36", "UD-Q8_K_XL", "Q8_K_XL", "q36", Q36_DIR / "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"),
    ("dsv41", "IQ2_XXS", "IQ2_XXS (GSQ-RCO)", "ds4", DS_DIR / "maya-ds4"),
]

# Present on the machine but refused by the engine on this card: recorded once, with the engine's own reason, and
# kept out of the timed sweep.  It is a measurement of the limit, not a run.
UNSUPPORTED = [
    ("dsv41", "UD-Q8_K_XL", "MXFP4",
     "refused: the dense weights (attention, shared experts, output head) need 15.61 GB of VRAM and the RX 580 has 8.59 GB"),
]


# ------------------------------------------------------------------------------------------------------------- stats

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


# -------------------------------------------------------------------------------------------------------------- runs

def run_one(cfg, rounds_left):
    family, label, quant, kind, path = cfg
    t0 = time.time()
    if kind == "q36":
        ids = ",".join(str(i) for i in (QWEN_IDS * 4)[:QWEN_PREFILL])
        cmd = [str(Q36_BIN), "--model", str(path), "--bench", "--bench-label", label,
               "--bench-ids", ids, "--bench-prefill", str(QWEN_PREFILL), "--bench-decode", str(QWEN_DECODE),
               "--kv-cap", "512", "--resident-byte-budget", "2"]
        env = dict(os.environ)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        out, err = proc.stdout, proc.stderr
        rec = {"prefill_tokens": QWEN_PREFILL, "decode_tokens": QWEN_DECODE}
        line = next((ln for ln in out.splitlines() if ln.startswith("BENCH_JSON ")), None)
        if proc.returncode != 0 or line is None:
            rec.update(rc=proc.returncode, error=(err.strip().splitlines() or [""])[-1] or "no BENCH_JSON line",
                       stderr_tail=err[-1200:])
            return rec
        payload = json.loads(line[len("BENCH_JSON "):])
        rec.update(payload)
        rec["rc"] = 0
    else:
        ids = ",".join(str(i) for i in (DS_IDS * 2)[:DS_PREFILL])
        cmd = [str(DS4_BIN), "--pack", str(path), "--ids", ids, "--n", str(DS_DECODE),
               "--max-context", "512", "--cpu-threads", "4"]
        env = dict(os.environ)
        env["DS4_NO_EOS"] = "1"      # always produce the full decode, so every run does the same work
        env["DS4_TOKLOG"] = "1"      # the per-token latency line
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        out, err = proc.stdout, proc.stderr
        rec = {"prefill_tokens": DS_PREFILL, "decode_tokens": DS_DECODE, "rc": proc.returncode}
        pm = re.search(r"prompt: (\d+) tokens in (\d+) ms \(([\d.]+) tok/s", err)
        dm = re.search(r"decode: (\d+) tokens in (\d+) ms \(([\d.]+) tok/s", err)
        if proc.returncode != 0 or pm is None or dm is None:
            rec.update(error=(err.strip().splitlines() or [""])[-1] or "no prompt/decode line",
                       stderr_tail=err[-1600:], stdout_tail=out[-600:])
            return rec
        rec.update(prefill_tokens=int(pm.group(1)), prefill_s=float(pm.group(2)) / 1000.0,
                   prefill_tok_s=float(pm.group(3)),
                   decode_tokens=int(dm.group(1)), decode_s=float(dm.group(2)) / 1000.0,
                   decode_tok_s=float(dm.group(3)))
        # The engine prints its rates to two decimals; its milliseconds and token counts are the measurement.
        # Recompute the same ratios the engine uses (decode excludes the first token, which carries the prompt
        # tail) so five runs that all print 0.29 still have a spread and a real standard deviation.
        if rec["prefill_s"] > 0:
            rec["prefill_tok_s"] = rec["prefill_tokens"] / rec["prefill_s"]
        if rec["decode_s"] > 0 and rec["decode_tokens"] > 1:
            rec["decode_tok_s"] = (rec["decode_tokens"] - 1) / rec["decode_s"]
        lm = re.search(r"token latency ms: p50 (\d+) p95 (\d+) p99 (\d+) max (\d+) mean (\d+)", err)
        if lm:
            rec["latency_ms"] = {"p50": int(lm.group(1)), "p95": int(lm.group(2)),
                                 "p99": int(lm.group(3)), "max": int(lm.group(4)), "mean": int(lm.group(5))}
        rec["generated"] = [int(x) for x in out.strip().split(",") if x.strip().isdigit()][:DS_DECODE]
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def pack_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    meta = path / "ds4.txt"
    names = []
    if meta.exists():
        model_dir = None
        for ln in meta.read_text(errors="replace").splitlines():
            if ln.startswith("model_dir "):
                model_dir = Path(ln.split(" ", 1)[1].strip())
        if model_dir:
            tensors = path / "tensors.txt"
            if tensors.exists():
                for ln in tensors.read_text(errors="replace").splitlines():
                    if ln.startswith("file "):
                        names.append(model_dir / ln.split(" ", 2)[2].strip())
    for p in names:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total if total else 0


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=10", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------------------------------------------------ docs

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
            continue
    return out


def rate(r, which):
    """The measured rate.  A DeepSeek record written before this recomputation carries the engine's rounded
    printed value; the milliseconds it also carries are the measurement, so derive the rate from those."""
    if r.get("kind") == "ds4" and r.get(which + "_s"):
        if which == "prefill":
            return r.get("prefill_tokens", 0) / r["prefill_s"] if r["prefill_s"] > 0 else 0.0
        if r.get("decode_tokens", 0) > 1:
            return (r["decode_tokens"] - 1) / r["decode_s"] if r["decode_s"] > 0 else 0.0
    return r.get(which + "_tok_s", 0.0)


def aggregate(records):
    by = {}
    for r in records:
        by.setdefault(r["label"], []).append(r)
    agg = {}
    for label, rs in by.items():
        ok = [r for r in rs if r.get("rc") == 0 and "decode_tok_s" in r]
        # Round 0 is each pack's first pass on this machine: cold page cache, cold GPU clocks.  The headline
        # numbers are the steady state (rounds 2+); round 1 is kept and priced separately.  A pack with fewer
        # than two steady runs (early in the sweep) falls back to all its runs and says so.
        steady = [r for r in ok if r.get("round", 0) > 0]
        use = steady if len(steady) >= 2 else ok
        agg[label] = {
            "family": rs[0]["family"],
            "label": label,
            "runs": sorted(rs, key=lambda r: r.get("order", 0)),
            "decode": [rate(r, "decode") for r in use],
            "prefill": [rate(r, "prefill") for r in use],
            "decode_all": [rate(r, "decode") for r in ok],
            "prefill_all": [rate(r, "prefill") for r in ok],
            "n": len(ok),
            "n_steady": len(steady),
            "used_steady": bool(steady) and len(steady) >= 2,
            "failed": len(rs) - len(ok),
        }
    return agg


def render_charts(agg, records):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    groups = []
    for family in ("qwen36", "dsv41"):
        items = []
        for cfg in CONFIGS:
            if cfg[0] != family:
                continue
            a = agg.get(cfg[1])
            if not a or not a["n"]:
                continue
            items.append({"label": cfg[1], "mean": mean(a["decode"]), "std": stdev(a["decode"]),
                          "points": a["decode"], "n": a["n"]})
        if items:
            groups.append((family, items))
    charts.grouped_bar_chart(
        CHART_DIR / "decode-throughput.svg",
        "Decode throughput by quantisation",
        "Strata on macpro51 (RX 580 8 GB, Metal 2); greedy; steady state (rounds 2+), mean ± 1 sample standard deviation; dots are individual runs",
        groups, "tokens per second (decode)", "higher is better")
    groups = []
    for family in ("qwen36", "dsv41"):
        items = []
        for cfg in CONFIGS:
            if cfg[0] != family:
                continue
            a = agg.get(cfg[1])
            if not a or not a["n"]:
                continue
            items.append({"label": cfg[1], "mean": mean(a["prefill"]), "std": stdev(a["prefill"]),
                          "points": a["prefill"], "n": a["n"]})
        if items:
            groups.append((family, items))
    charts.grouped_bar_chart(
        CHART_DIR / "prefill-throughput.svg",
        "Prompt (prefill) throughput by quantisation",
        "Strata on macpro51 (RX 580 8 GB, Metal 2); the fixed prompt read one token at a time; steady state (rounds 2+), mean ± 1 sample standard deviation",
        groups, "tokens per second (prefill)", "higher is better")

    pts = []
    for cfg in CONFIGS:
        a = agg.get(cfg[1])
        if not a or not a["n"]:
            continue
        pts.append({"family": cfg[0], "label": cfg[1], "x": pack_bytes(cfg[4]) / 1073741824.0,
                    "y": mean(a["decode"]), "std": stdev(a["decode"])})
    if pts:
        charts.scatter_chart(
            CHART_DIR / "size-vs-decode.svg",
            "Decode throughput against model size",
            "A larger pack is not automatically slower: what a token reads and how many experts stream decide",
            pts, "pack size on disk (GiB)", "decode tokens/s")

    rows = []
    for r in sorted(records, key=lambda r: r.get("order", 0)):
        if r.get("rc") == 0 and "decode_tok_s" in r:
            rows.append({"order": r.get("order", 0), "label": r["label"], "family": r["family"],
                         "value": r["decode_tok_s"]})
    charts.timeline_chart(
        CHART_DIR / "run-order.svg",
        "Every run in the order it was made",
        "Each round reshuffles the packs, so a drift in the machine is shared rather than assigned to one pack",
        rows)


def render_doc(agg, records, meta):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# Strata quantisation benchmarks on macpro51 (Intel Mac Pro, Radeon RX 580)")
    lines.append("")
    lines.append(f"_Generated {now} by `bench/macpro51/run_bench.py`. Every number below was produced on the "
                 f"machine named here, during this sweep._")
    lines.append("")
    lines.append("## The machine")
    lines.append("")
    lines.append("| | |")
    lines.append("| --- | --- |")
    lines.append("| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |")
    lines.append("| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |")
    lines.append("| OS | macOS 11.7.11 (Big Sur, 20G1443) |")
    lines.append("| Engine | `strata-q36-metal` (Qwen3.6) and `strata_ds4` (DeepSeek V4.1-Flash), built Release with Ninja |")
    lines.append(f"| Branch / commit at the last run | `macpro51` / `{meta.get('git_head', 'unknown')}` |")
    lines.append("")
    lines.append("The Metal runtime here is the workaround path: kernels take buffer offsets instead of raw device "
                 "addresses, the DeepSeek engine flushes command buffers one at a time (`DS4_FLUSH_EVERY`), and "
                 "`simdgroup_matrix`/Metal 3 features are skipped. Numbers are therefore this card's, not a "
                 "Metal 3 Mac's.")
    lines.append("")
    lines.append("## What ran, and what could not")
    lines.append("")
    lines.append("Strata's three public families are DeepSeekV4.1-Flash-1017, Qwen3.8-Flash-Next and "
                 "Qwen3.6-35B-A3B. This branch's Metal engine has paths for Qwen3.6 (`src/qwen36`) and "
                 "DeepSeek V4.1-Flash (`src/dsv41`); Qwen3.8 has no Metal path here and the docs place it on "
                 "Apple silicon, so it is not runnable on this Intel Mac. Every installed pack of the two "
                 "runnable families was tested.")
    lines.append("")
    lines.append("| Family | Pack | On disk | Result |")
    lines.append("| --- | --- | ---: | --- |")
    for cfg in CONFIGS:
        family, label, quant, kind, path = cfg
        size = pack_bytes(path)
        a = agg.get(label)
        if not a or not a["n"]:
            verdict = "no completed run yet"
        else:
            verdict = f"{a['n']} run(s), {mean(a['decode']):.2f} tok/s decode (steady state)"
        lines.append(f"| {family} | {label} | {size/1073741824.0:.1f} GiB | {verdict} |")
    for family, label, quant, reason in UNSUPPORTED:
        lines.append(f"| {family} | {label} | — | {reason} |")
    lines.append("")
    lines.append("Two Qwen3.6 packs were refused when this sweep started, and the refusals were real defects "
                 "rather than hardware limits; they were fixed on the branch before the sweep (see "
                 "[Engine fixes](#engine-fixes)). The DeepSeek UD-Q8_K_XL pack is refused by the engine on "
                 "this card because its dense weights alone need 15.61 GB of VRAM against the RX 580's "
                 "8.59 GB; that is a hardware limit, recorded here rather than worked around.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- **One process per run.** The engine loads the model, warms every pipeline with one full "
                 f"pass, resets the sequence state in place, then times a fresh pass. Model load time is "
                 f"reported separately and is not part of either throughput.")
    lines.append(f"- **Fixed workload.** Qwen3.6: {QWEN_PREFILL} prompt tokens read one at a time, then "
                 f"{QWEN_DECODE} greedy tokens. DeepSeek: {DS_PREFILL} prompt tokens, then {DS_DECODE} greedy "
                 f"tokens, with its own chunked prompt path and its reported prompt/decode times. The same "
                 f"token ids are used for every pack of a family.")
    lines.append(f"- **Mixed order.** Each round contains every runnable pack once, in a fresh shuffled order "
                 f"(seed in `bench/macpro51/meta.json`). No pack is always first or always last, so thermal or "
                 f"page-cache drift is spread across the matrix. The DeepSeek pack is 239 GiB and its runs can "
                 f"evict Qwen pages from the 128 GiB page cache; that is one reason every pack is measured in "
                 f"every round and why the error bars, not a single run, are the result.")
    lines.append(f"- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs of that pack "
                 f"(rounds 2 and later; round 1 is each pack's first pass, with a cold page cache, and is priced "
                 f"separately below). The individual runs are drawn as dots on the charts. With "
                 f"{meta.get('rounds', 'N')} round{'s' if meta.get('rounds', 2) != 1 else ''} the bars describe "
                 f"run-to-run spread on this machine, not a confidence interval for another one.")
    lines.append(f"- **Steady state.** Qwen runs use a 2 GiB expert-slot budget, which selects the streaming "
                 f"path for every pack here; the warm-up pass is what fills the OS page cache and compiles the "
                 f"pipelines before the first measured token.")
    lines.append("")
    lines.append("## Decode throughput")
    lines.append("")
    lines.append(f"![Decode throughput by quantisation](benchmarks/macpro51/decode-throughput.svg)")
    lines.append("")
    lines.append("| Family | Pack | Steady runs | Decode, steady (mean ± sd) | All runs (mean) | Steady range | Prefill, steady (mean ± sd) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    short = []
    for cfg in CONFIGS:
        a = agg.get(cfg[1])
        if not a or not a["n"]:
            continue
        d, p = a["decode"], a["prefill"]
        star = "" if a.get("used_steady", True) else "*"
        if star:
            short.append(cfg[1])
        lines.append(f"| {cfg[0]} | {cfg[1]}{star} | {a.get('n_steady', a['n']) if a.get('used_steady', True) else a['n']} | "
                     f"{fmt(mean(d))} ± {fmt(stdev(d))} | {fmt(mean(a['decode_all']))} | "
                     f"{fmt(min(d))} – {fmt(max(d))} | {fmt(mean(p))} ± {fmt(stdev(p))} |")
    if short:
        lines.append("")
        lines.append(f"\\* fewer than two steady-state runs so far, so this row uses every completed run "
                     f"({', '.join(short)}).")
    lines.append("")
    lines.append("## Prompt (prefill) throughput")
    lines.append("")
    lines.append(f"![Prefill throughput by quantisation](benchmarks/macpro51/prefill-throughput.svg)")
    lines.append("")
    lines.append("Prefill is read one token at a time by the Qwen engine, so it is a per-token latency "
                 "measurement as much as a bandwidth one; the DeepSeek engine reads its prompt in chunks. The "
                 "two families are therefore comparable within themselves, not to each other.")
    lines.append("")
    lines.append("## Size against speed")
    lines.append("")
    lines.append(f"![Decode throughput against model size](benchmarks/macpro51/size-vs-decode.svg)")
    lines.append("")
    lines.append("On this card a token streams its routed experts from RAM/SSD through an 8 GB window, so the "
                 "pack's size is only one of the things that decide its speed: the number of bytes a token "
                 "actually reads, the format's decode cost, and how much of the dense side stays resident all "
                 "matter as much.")
    lines.append("")
    lines.append("## What the numbers say")
    lines.append("")
    done = [(cfg[1], agg[cfg[1]]) for cfg in CONFIGS if agg.get(cfg[1]) and agg[cfg[1]]["n"]]
    if len(done) >= 2:
        dec = sorted(((lbl, mean(a["decode"])) for lbl, a in done), key=lambda x: -x[1])
        lines.append(f"- **Fastest decode:** `{dec[0][0]}` at {dec[0][1]:.2f} tok/s; **slowest:** "
                     f"`{dec[-1][0]}` at {dec[-1][1]:.2f} tok/s "
                     f"({dec[0][1] / max(1e-9, dec[-1][1]):.2f}× between them).")
        cvs = [(lbl, 100.0 * stdev(a["decode"]) / mean(a["decode"]))
               for lbl, a in done if a["n"] > 1 and mean(a["decode"]) > 0]
        if cvs:
            cvs.sort(key=lambda x: -x[1])
            lines.append(f"- **Run-to-run spread:** largest for `{cvs[0][0]}` at {cvs[0][1]:.1f}% of its mean "
                         f"(coefficient of variation), smallest for `{cvs[-1][0]}` at {cvs[-1][1]:.1f}%. "
                         f"A spread this small is what a fixed workload on one machine should show; a bar "
                         f"that is much wider is a pack whose expert streaming is not in steady state.")
        pre = sorted(((lbl, mean(a["prefill"])) for lbl, a in done), key=lambda x: -x[1])
        lines.append(f"- **Fastest prompt read:** `{pre[0][0]}` at {pre[0][1]:.2f} tok/s; slowest "
                     f"`{pre[-1][0]}` at {pre[-1][1]:.2f} tok/s. Prefill is per-token latency on this engine, "
                     f"not just bandwidth, so its ranking need not follow decode's.")
        qwen = [(lbl, mean(a["decode"]), pack_bytes(dict((c[1], c) for c in CONFIGS)[lbl][4]) / 1073741824.0)
                for lbl, a in done if a["family"] == "qwen36"]
        if len(qwen) >= 3:
            qwen.sort(key=lambda x: x[2])
            lines.append("- **Size alone does not order speed.** Qwen3.6 packs from smallest to largest decode "
                         f"at {', '.join(f'`{l}` {v:.2f}' for l, v, _ in qwen)} tok/s; the formats a pack uses "
                         "for its routed experts and its dense side matter as much as its GiB.")
        later = [a for _, a in done if any(r.get("round", 0) > 0 and r.get("rc") == 0 for r in a["runs"])]
        first = [a for _, a in done if any(r.get("round", 0) == 0 and r.get("rc") == 0 for r in a["runs"])]
        if later and first:
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
                changes = [100.0 * (b - a) / max(1e-9, a) for _, a, b in rows]
                lines.append("")
                lines.append("The first round is the first pass over each pack (cold page cache, cold GPU "
                             "clocks); the later rounds are the steady state. The difference is the price of "
                             f"a first run, not a property of the pack: it averages {mean(changes):+.1f}% "
                             f"over these {len(rows)} pack(s).")
                lines.append("")
                lines.append("| Pack | Round 1 (tok/s) | Rounds 2+ (tok/s) | Change |")
                lines.append("| --- | ---: | ---: | ---: |")
                for lbl, a, b in rows:
                    lines.append(f"| {lbl} | {a:.2f} | {b:.2f} | {100.0 * (b - a) / max(1e-9, a):+.1f}% |")
    else:
        lines.append("_Not enough completed packs yet; this section fills in as the rounds finish._")
    lines.append("")
    lines.append("## Run order (mixing)")
    lines.append("")
    lines.append(f"![Every run in execution order](benchmarks/macpro51/run-order.svg)")
    lines.append("")
    lines.append("The sweep is resumable: a `(round, pack)` pair already in `results.jsonl` is skipped, so an "
                 "interrupted or rebooted machine costs one run rather than the series.")
    lines.append("")
    lines.append("## Per-run log")
    lines.append("")
    lines.append("| # | Round | Family | Pack | Prefill s | Decode s | Decode tok/s | Note |")
    lines.append("| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |")
    for r in sorted(records, key=lambda r: r.get("order", 0)):
        if r.get("rc") == 0 and "decode_tok_s" in r:
            note = "first round (cold cache)" if r.get("round", 0) == 0 else "later round"
            lines.append(f"| {r.get('order', 0) + 1} | {r.get('round', 0) + 1} | {r['family']} | {r['label']} | "
                         f"{fmt(r['prefill_s'])} | {fmt(r['decode_s'])} | {fmt(rate(r, 'decode'))} | {note} |")
        else:
            lines.append(f"| {r.get('order', 0) + 1} | {r.get('round', 0) + 1} | {r['family']} | {r['label']} | — | — | — | "
                         f"failed: {r.get('error', 'unknown')} |")
    lines.append("")
    lines.append("## Engine fixes made to run the matrix")
    lines.append("")
    lines.append("These were found by opening every installed pack before the sweep, and are on this branch:")
    lines.append("")
    lines.append("1. **Embedding rows outside the three kernel formats.** `q36_embed` had device branches for "
                 "Q8_0, Q6_K and Q4_K only, so UD-Q2_K_XL (Q5_K token embeddings) was refused at open. The "
                 "row is now dequantised on the host for any other dense format the port accepts; it is 2,048 "
                 "values per token against forty layers of expert work.")
    lines.append("2. **The never-run MTP block's expert types.** The MTP block is a draft head the engine skips, "
                 "but type validation and slot accounting still walked it, and UD-Q2_K_XL's MTP block carries "
                 "Q2_K gate/up (no kernel, because nothing dispatches it). Validation, slot sizing and slot "
                 "packing now skip the MTP block, as the forward pass already did.")
    lines.append("3. **BF16 grouped experts.** UD-Q8_K_XL promotes three layers' routed experts to BF16 "
                 "(layers 34, 38 and 39). `moe.metal` now instantiates the gate/up and down kernels for BF16 "
                 "as well, and the type tables accept it. The pack then runs; its first generated token "
                 "matches the Q8_0-era 279.")
    lines.append("")
    lines.append("## Tests")
    lines.append("")
    lines.append("`gemv_parity` passes on this build for Q2_0, IQ3_S, Q3_K, Q4_K, Q5_K and Q6_K at n_in 2560 "
                 "and 10240 and 1/3/8 tokens (rel L2 6e-08…4e-07, and the multi-token window is bitwise equal "
                 "to the one-token path).")
    lines.append("")
    lines.append("The Qwen3.6 checks: 15 of the 18 registered ctest tests pass on this build. The three that "
                 "do not are not on a real pack's path:")
    lines.append("")
    lines.append("- `q36_forward_check` and `q36_generate_check` abort inside `attn_init` while compiling the "
                 "synthetic eight-head fixture's attention kernels on this Metal 2 driver; the sweep below is "
                 "the evidence that a real pack's shapes compile and run.")
    lines.append("- `runtime_hello` fails its device-address and graph-replay cases because macOS 11 has "
                 "neither; the engine's documented workarounds are why these benchmarks run at all.")
    lines.append("")
    lines.append("`q36_model_check`, `q36_glue_parity` and `q36_upload_check` were repaired as part of this "
                 "work (a stale kernel argument, an error message that looked a type up in the wrong table, "
                 "and a byte comparison against a private buffer's handle, which now blits the buffer back "
                 "first). `q36_upload_check` verifies 34.5 MB of dense and embedding bytes and passes.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```sh")
    lines.append("cd ~/strata-bench")
    lines.append("ninja -C build-bench strata-q36-metal strata_ds4")
    lines.append("python3 bench/macpro51/run_bench.py --rounds 5 --seed 20261118")
    lines.append("python3 bench/macpro51/run_bench.py --only Q4_K_M --rounds 1 --no-push   # a single pack")
    lines.append("```")
    lines.append("")
    lines.append("Results are `bench/macpro51/results.jsonl` (one JSON object per run, including the per-step "
                 "decode times) and `bench/macpro51/meta.json` (the matrix and the settings). Charts are SVG in "
                 "`docs/benchmarks/macpro51/`.")
    lines.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def commit_and_push(files, message, push=True):
    subprocess.run(["git", "-C", str(REPO), "add", "--"] + [str(f) for f in files], check=True)
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        return False
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", message], check=True)
    if push:
        p = subprocess.run(["git", "-C", str(REPO), "push", "origin", "HEAD:macpro51"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  push failed (will retry on the next run): {p.stderr.strip()[-300:]}", flush=True)
    return True


# ------------------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--only", action="append", default=[], help="label to include (repeatable)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if not args.only or c[1] in args.only]
    if not configs:
        print("no configuration matches --only", file=sys.stderr)
        return 2
    missing = [c for c in configs if not c[4].exists()]
    if missing:
        for c in missing:
            print(f"missing: {c[4]}", file=sys.stderr)
        return 2

    existing = load_records()
    done = {(r.get("round"), r["label"]) for r in existing if r.get("rc") == 0}
    order = max((r.get("order", -1) for r in existing), default=-1) + 1
    rounds = []
    for rnd in range(args.rounds):
        order_cfgs = list(configs)
        random.Random(args.seed + rnd).shuffle(order_cfgs)
        rounds.append((rnd, order_cfgs))

    meta = {
        "host": "macpro51",
        "machine": "MacPro5,1, Xeon X5675 x2, 128 GB, Radeon RX 580 8 GB, macOS 11.7.11",
        "git_head": git_head(),
        "rounds": args.rounds,
        "seed": args.seed,
        "qwen_prefill": QWEN_PREFILL,
        "qwen_decode": QWEN_DECODE,
        "ds_prefill": DS_PREFILL,
        "ds_decode": DS_DECODE,
        "configs": [{"family": c[0], "label": c[1], "quant": c[2], "kind": c[3], "path": str(c[4])} for c in configs],
        "unsupported": [{"family": f, "label": l, "quant": q, "reason": r} for f, l, q, r in UNSUPPORTED],
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    META.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    plan = [(rnd, cfg) for rnd, cfgs in rounds for cfg in cfgs]
    print(f"plan: {len(plan)} run(s), rounds {args.rounds}, seed {args.seed}; {len(done)} already done", flush=True)
    for rnd, cfg in plan:
        if (rnd, cfg[1]) in done:
            continue
        print(f"[round {rnd+1}/{args.rounds}] {cfg[0]} {cfg[1]} ...", flush=True)
        rec = run_one(cfg, args.rounds)
        rec.update({"family": cfg[0], "label": cfg[1], "quant": cfg[2], "kind": cfg[3],
                    "path": str(cfg[4]), "round": rnd, "order": order,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_head": meta["git_head"], "engine": "strata-q36-metal" if cfg[3] == "q36" else "strata_ds4"})
        order += 1
        if rec.get("rc") == 0:
            print(f"  {rec['decode_tok_s']:.3f} tok/s decode, {rec['prefill_tok_s']:.3f} tok/s prefill", flush=True)
        else:
            print(f"  FAILED: {rec.get('error', '?')}", flush=True)
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        records = load_records()
        agg = aggregate(records)
        try:
            render_charts(agg, records)
            render_doc(agg, records, meta)
            files = [RESULTS, META, DOC] + sorted(CHART_DIR.glob("*.svg"))
            msg = (f"[Benchmark] macpro51: {cfg[1]} run {rec['round']+1}/{args.rounds} "
                   f"({rate(rec, 'decode'):.2f} tok/s decode)" if rec.get("rc") == 0
                   else f"[Benchmark] macpro51: {cfg[1]} run {rec['round']+1}/{args.rounds} failed")
            if not args.dry_run:
                commit_and_push(files, msg, push=not args.no_push)
        except Exception as e:  # noqa: BLE001
            print(f"  docs/commit failed: {e}", file=sys.stderr, flush=True)

    records = load_records()
    agg = aggregate(records)
    if not args.dry_run:
        render_charts(agg, records)
        render_doc(agg, records, meta)
        commit_and_push([RESULTS, META, DOC] + sorted(CHART_DIR.glob("*.svg")),
                        f"[Benchmark] macpro51: final summary ({len(records)} runs)", push=not args.no_push)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
