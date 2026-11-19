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

# Packs the Metal kernels of this branch cannot run, with the engine's own message (recorded, never timed).
REFUSED = [
    ("IQ2_XS", "the window has no kernel for its IQ1_M gate/up experts: `mtl: no function moe_gu_29_32_1_4`"),
    # Two packs whose gate/up experts are IQ2_XXS (16) return the same wrong answer; the kernels pass
    # expert_parity on their real data, so the engine's type-16 path (x planes/sums binding, plan, arena) is the
    # suspect, not the decoder.
    ("UD-Q2_0", "its greedy answer is `76 76 1544 7509 ...` where every other pack gives the reference tokens; its "
                "gate/up experts are IQ2_XXS (16), the same type as UD-IQ1_M's, and `expert_parity` passes on this "
                "pack's real layer-0 data at every GU tile"),
    # The v3.1 sets answer differently: UD-IQ2_XS joins UD-Q2_0 and UD-IQ1_M (same divergent first tokens, same
    # v3.1 revision).  Their expert formats are all types that the answering IQ3_XXS/IQ3_S packs also use, so the
    # matrix records them as refused on the answer gate, not as slow quantisations.
    ("UD-IQ2_XS", "its greedy answer is `4960 4654 1544 159578 ...`, not the reference tokens; a v3.1 set like "
                  "UD-Q2_0 and UD-IQ1_M, whose expert formats are all used by the answering IQ3_XXS and IQ3_S "
                  "packs"),
    # UD-Q2_K_XL (Unsloth v3.1, 2-bit K-quant dynamic) joins the v3.1 refusals.
    ("UD-Q2_K_XL", "its greedy answer is `76 76 1544 7509 76 76 7509 88`, the v3.1 signature, where every working "
                   "pack gives the reference tokens; its expert formats are all in the answering packs' set"),
    # Gyro (agentionai, trellis experts) runs but does not answer either: the TQ path executes without a missing
    # kernel, and its greedy output diverges (`198 18 18 ...` for Gyro-S's own smoke; the sweep runs are similar).
    ("Gyro-S", "runs through the trellis expert path (no missing kernel) but its greedy answer does not match the "
                "reference: `198 18 18 18 18 18 18 18` in its own smoke, and the sweep runs diverge as well"),
    ("Gyro-M", "same: the TQ path runs and its greedy answer diverges from the reference"),
    # UD-IQ1_M runs but does not answer: the same prompt gives 76 76 1544 7509 ... where every pack that uses
    # Q2_0 / IQ3_XXS / IQ3_S / Q4_K-Q6_K experts gives 271 248068 198 760 1156 6587 264 61446, at every GU tile
    # (32,1,4 / 32,2,4 / 16,2,4 / 8,1,4).  Its gate/up type is T_IQ2_XXS (16), which no other measured pack uses.
    ("UD-IQ1_M", "runs, but its forward is degenerate (`76 76 1544 7509 4654 76 88 76`, first-token logits at "
                 "cosine 0.16 to Q2_0's); the residual is healthy after layer 0 (0.177; Q2_0 0.181), still healthy "
                 "after layer 1's attention (0.173; Q2_0 0.174, UD-Q4 0.170), and 15.1 after layer 1's expert "
                 "dispatch (UD-Q4 0.348), then stays there.  The PLE block's own outputs (keyn/qn/gated/normW) match "
                 "the healthy packs to three digits, the experts pass `expert_parity` on this pack's real data (rel "
                 "L2 1.3e-7), and layer 1 uses the same expert types as layer 0 - so the fault is inside layer 1's "
                 "routed/shared expert path (plan, gu, swiglu, down, combine), not the quantisation"),
]

PREFILL, DECODE = 32, 32
# The same prompt gives the same first tokens on every pack that answers (measured on Q2_0, IQ3_XXS, IQ3_S, UD-Q4,
# UD-Q5 and UD-Q6_K_XL).  A pack whose greedy output starts elsewhere is not answering, whatever its rate, so it is
# not reported as a slow quantisation.
EXPECT = [271, 248068, 198, 760]
PREFILL_ONLY = False   # --prefill-only: one token after the prompt, so the prompt is measured and no decode rate is claimed
# Qwen3.6's ids (the same 248 320-entry Qwen vocabulary): the first 32 of
#   "Explain, in two short sentences, why the sky looks blue during the day and red at sunset. Then say
#    why the Moon looks orange near the horizon."
# The pack's own tokenizer, not Qwen3.6's ids: the two vocabularies share a size but not their order, and the
# Qwen3.6 ids render as punctuation here.  Encoded from the sentence below with tools/strata_tokenizer.py (32 ids).
IDS = [814, 20139, 11, 303, 1330, 2716, 22157, 11, 3069, 279, 12515, 5686, 6105, 2261, 279, 1834, 321,
       2438, 506, 41564, 13, 4844, 1910, 3069, 279, 16820, 5686, 18016, 3043, 279, 32946, 13]

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
    ("UD-IQ3_XXS", "UD-IQ3_XXS", "UD-IQ3_XXS (Unsloth v3.1)", "unsloth", "3-bit i-quant dynamic"),
    ("UD-Q3_K_XL", "UD-Q3_K_XL", "UD-Q3_K_XL (Unsloth v3.1)", "unsloth", "3-bit K-quant dynamic"),
    ("UD-IQ4_XS", "UD-IQ4_XS", "UD-IQ4_XS (Unsloth v3.1)", "unsloth", "~4-bit i-quant experts"),
    ("UD-Q4_K_XL", "UD-Q4_K_XL", "UD-Q4_K_XL (Unsloth)", "unsloth", "4-bit K-quant experts"),
    ("UD-Q5_K_XL", "UD-Q5_K_XL", "UD-Q5_K_XL (Unsloth)", "unsloth", "5-bit K-quant experts"),
    ("UD-Q6_K_XL", "UD-Q6_K_XL", "UD-Q6_K_XL (Unsloth)", "unsloth", "6-bit K-quant experts"),
    ("Q8_0", "Q8_0", "Q8_0 (Unsloth)", "unsloth", "8-bit, the reference quality"),
    # Gyro: one GGUF each, the n-gram table inside the file, trellis experts (TQ2_T / TQK6 / TQK7), no repack
    ("Gyro-S", "Gyro-S", "Gyro-S (TQ1_0, agentionai)", "agentionai", "1.58-bit trellis experts"),
    ("Gyro-M", "Gyro-M", "Gyro-M (TQ2_0, agentionai)", "agentionai", "2-bit trellis experts"),
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


def repacked(label: str):
    """A multi-shard pack repacked into the one GGUF the Metal window opens, or None."""
    d = pack_dir(label)
    f = d / f"{label}-metal.gguf"
    return f if f.is_file() else None


def ple_shard(label: str):
    """The shard that holds the PLE n-gram table (the repack leaves it there; the window reads it through
    --ple-gguf).  The last shard in these sets, found by its tensor name rather than assumed."""
    d = pack_dir(label)
    cands = sorted(d.glob("*-of-*.gguf"), reverse=True)
    for p in cands:
        try:
            with open(p, "rb") as f:
                if b"per_layer_token_embd.weight" in f.read(24 << 20):
                    return p
        except OSError:
            continue
    return None


def native_pair(label: str):
    """(native shard/file for --native, PLE shard for --ple-gguf)."""
    rep = repacked(label)
    if rep is not None:
        return rep, ple_shard(label)
    one, two = shards(label)
    return one, two


def ready(label: str) -> bool:
    """A pack is ready when it can actually be opened: a repacked single file with its metadata pack, or a
    complete two-shard ISTA pack with its metadata pack.  A download in progress is not ready."""
    d = pack_dir(label)
    if not d.is_dir():
        return False
    if repacked(label) is not None:
        return (d / "pack" / "native_experts.txt").exists()
    if len(list(d.glob("*-of-*.gguf"))) > 2:
        return False          # a multi-shard set without its repacked single file
    one, two = shards(label)
    return one is not None and two is not None and (d / "pack" / "native_experts.txt").exists()


def pack_size(label: str) -> int:
    """The pack's download size (a repacked copy beside it is a working file, not part of the pack)."""
    d = pack_dir(label)
    if not d.is_dir():
        return 0
    return sum(p.stat().st_size for p in d.glob("*.gguf") if not p.name.endswith("-metal.gguf"))


QTYPE = {2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
         23: "IQ4_XS", 29: "IQ1_M", 42: "Q2_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7"}


def pack_types(label):
    """The pack's routed-expert formats, as counts of layers: [(gate/up, down, layers), ...].

    Read from native_experts.txt (the pack metadata), so it describes the download, not a run."""
    f = pack_dir(label) / "pack" / "native_experts.txt"
    if not f.is_file():
        return []
    seen = {}
    try:
        with open(f) as fh:
            for line in fh:
                if not line or line[0] in "#\n":
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                key = (int(parts[1]), int(parts[2]))
                seen[key] = seen.get(key, 0) + 1
    except (OSError, ValueError):
        return []
    return [(QTYPE.get(g, str(g)), QTYPE.get(d, str(d)), n) for (g, d), n in sorted(seen.items(), key=lambda kv: -kv[1])]


def types_text(label):
    tt = pack_types(label)
    if not tt:
        return "—"
    return ", ".join(f"{g} / {d} ({n})" for g, d, n in tt)


def blob_mb(label):
    """The largest expert blob (MB) of a prepared pack: the maximum blob_bytes over its native_experts.txt.

    This is the number the engine prints as "largest blob", computed from the pack metadata instead of parsed
    out of a run, so old records get it too."""
    f = pack_dir(label) / "pack" / "native_experts.txt"
    if not f.is_file():
        return None
    best = 0
    try:
        with open(f) as fh:
            for line in fh:
                if not line or line[0] in "#\n":
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    best = max(best, int(parts[4]))
    except (OSError, ValueError):
        return None
    return best / 1e6 if best else None


def blob_of(a):
    """The pack's largest expert blob (MB): a recorded value, else the pack metadata."""
    for r in reversed(a.get("runs", [])):
        if r.get("blob_mb"):
            return float(r["blob_mb"])
    for r in a.get("runs", []):
        b = blob_mb(r.get("label", "")) if r.get("label") else None
        if b:
            return b
    return None


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
    one, _ = native_pair(label)
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
    one, two = native_pair(label)
    if one is None or two is None:
        rec.update(rc=-1, error="not downloaded (or, for a multi-shard pack, not repacked)")
        return rec
    prepared = prepare_pack(label)
    rec["pack_state"] = prepared
    rec["blob_mb"] = blob_mb(label)
    if prepared != "ready":
        rec.update(rc=-2, error=prepared)
        return rec
    ids = ",".join(str(i) for i in IDS[:PREFILL])
    decode_n = 1 if PREFILL_ONLY else DECODE
    # One token per verify window, no drafter: every pack runs the same policy, and the window path is the
    # quantisation's own cost.  The prompt goes through the engine's prompt path in one chunk of PREFILL.
    cmd = [str(BIN), "--pack", str(pack_dir(label) / "pack"), "--native", str(one), "--ple-gguf", str(two)]
    if repacked(label) is not None or not list(pack_dir(label).glob("*-of-*.gguf")):
        # a repacked file (or a single-file pack such as Gyro's) holds every dense tensor; naming it as the dense
        # source keeps the PLE file out of the dense list (the split check refuses a repacked file beside a shard)
        cmd += ["--native-dense-gguf", str(one)]
    cmd += ["--tokens", ids, "--max-new", str(decode_n),
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
    # the bytes one routed expert of this pack costs (the pack line names its largest blob): the variable that
    # should explain a token's cost better than the file size
    bm = re.search(r"experts \(largest blob ([\d.]+) MB\)", err)
    if bm:
        rec["blob_mb"] = float(bm.group(1))
    if proc.returncode != 0 or dm is None:
        lines = [ln.strip() for ln in err.strip().splitlines() if ln.strip()]
        pick = next((ln for ln in lines if "no function " in ln or "no Metal implementation" in ln or
                     ln.startswith("strata generate:") or ln.startswith("strata: ")), None)
        rec.update(error=pick or (lines[-1] if lines else "no decode line"),
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
        rec["answer_ok"] = rec["generated"][:len(EXPECT)] == EXPECT   # (after the tokens: an early field was always False)
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
    def answers(r):
        """A run counts only if its greedy output is the reference answer.

        The tokens are the evidence; a stored answer_ok is only a fallback, because an early build wrote it before the
        tokens were parsed (always False, which silently emptied the matrix)."""
        gen = r.get("generated") or []
        if gen:
            return gen[:len(EXPECT)] == EXPECT
        return bool(r.get("answer_ok", False))
    for label, rs in by.items():
        ok = [r for r in rs if r.get("rc") == 0 and r.get("prefill_s") and answers(r)]
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


    bpts = []
    for c in CONFIGS:
        a = agg.get(c[1])
        if not a or not a["n_decode"]:
            continue
        b = blob_of(a)
        if b:
            bpts.append({"family": "qwen38", "label": c[1], "x": b,
                         "y": mean(a["decode"]), "std": stdev(a["decode"])})
    if bpts:
        charts.scatter_chart(
            CHART_DIR / "qwen38-decode-vs-blob.svg",
            "Qwen3.8-Flash-Next: decode throughput against the bytes a token reads",
            "One routed expert's blob is what a token pulls from the SSD ten times a layer: the closer two packs' blobs, the closer their rate",
            bpts, "largest expert blob (MB)", "decode tokens/s")

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
    starred = False
    missing = []
    for cfg in CONFIGS:
        size = pack_size(cfg[0])
        if not size:
            missing.append(cfg[1])
            continue
        a = agg.get(cfg[1])
        if not a or not a["n"]:
            L.append(f"| `{cfg[1]}` | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | — | — | — | — |")
            continue
        p = a["prefill"]
        star = "" if a.get("used_steady") else "*"
        starred = starred or bool(star)
        L.append(f"| `{cfg[1]}`{star} | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | {a['n']} | "
                 f"{fmt(mean(p))} ± {fmt(stdev(p))} | {fmt(mean(a['prefill_all']))} | {fmt(min(p))} – {fmt(max(p))} |")
    if starred:
        L.append("")
        L.append("\\* fewer than two steady-state runs (rounds 2+) so far, so this row uses every completed run.")
    if missing:
        L.append("")
        L.append("Not on disk yet (not attempted, and not counted as a failure): " + ", ".join(f"`{m}`" for m in missing) + ".")
    if REFUSED:
        L.append("")
        L.append("Refused on this branch (recorded, not timed):")
        L.append("")
        L.append("| Pack | Why |")
        L.append("| --- | --- |")
        for name, why in REFUSED:
            L.append(f"| `{name}` | {why} |")
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
    missing = []
    refused_names = {n for n, _ in REFUSED}
    for cfg in CONFIGS:
        label = cfg[1]
        if label in refused_names:
            continue
        a = agg.get(label)
        size = pack_size(cfg[0])
        if not size:
            missing.append(label)
            continue
        if a and a["n"]:
            verdict = f"{a['n']} run{'s' if a['n'] != 1 else ''}, {mean(a['decode']):.2f} tok/s decode (steady state)"
        elif a and [r for r in a["runs"] if r.get("rc") == 0]:
            verdict = "runs, but its greedy answer diverges from every other pack's"
        else:
            verdict = "queued"
        L.append(f"| `{label}` | {cfg[2]} | {cfg[3]} | {size/1073741824.0:.1f} GiB | {verdict} |")
    for name, why in REFUSED:
        L.append(f"| `{name}` | refused on this branch | — | {pack_size(name)/1073741824.0:.1f} GiB | {why} |")
    if missing:
        L.append("")
        L.append("Not on disk yet (not attempted, and not counted as a failure): " + ", ".join(f"`{m}`" for m in missing) + ".")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- **One process per run.** `strata` opens the shard, builds the pack metadata once, reads the dense side and the expert arena, then times the prompt and the decode separately. Load time is reported on its own and is not part of either throughput.")
    L.append(f"- **Fixed workload.** The same {PREFILL} token ids for every pack (the Qwen 248 320-entry vocabulary, a varied English sentence rather than repeated ids, so routing is not narrowed), read one at a time, then {DECODE} greedy tokens.")
    L.append(f"- **Mixed order.** Each round contains every downloaded pack once, in a fresh shuffled order (seed in the run's `meta.json`). No pack is always first or last, so thermal drift, page-cache state and SSD behaviour are spread across the matrix.")
    L.append(f"- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs (rounds 2+); round 1 is each pack's first pass and is priced separately. Individual runs are the dots on the charts.")
    L.append(f"- **Streaming.** `STRATA_METAL_STREAM=1` is set for every run: the pack cannot be resident in 8 GB of VRAM, and forcing the streamed path keeps every pack on the same code path.")
    L.append(f"- **The answer gate.** Every pack that works continues the prompt with the same first tokens "
             f"(`{' '.join(str(t) for t in EXPECT)}`). A run whose greedy output starts elsewhere is not a slower "
             f"quantisation, it is a pack that is not answering: it is excluded from the tables and marked in the "
             f"log, because a rate for a wrong forward is worse than no rate.")
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
    blob_rows = [(c[1], agg[c[1]], blob_of(agg[c[1]])) for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n_decode"]]
    blob_rows = [x for x in blob_rows if x[2]]
    if blob_rows:
        L.append("## The bytes a token reads")
        L.append("")
        L.append("![Decode throughput against the largest expert blob](benchmarks/macpro51/qwen38-decode-vs-blob.svg)")
        L.append("")
        L.append("| Pack | Largest expert blob | Decode, steady |")
        L.append("| --- | ---: | ---: |")
        for label, a, b in blob_rows:
            L.append("| `%s` | %.2f MB | %s \u00b1 %s tok/s |" % (label, b, fmt(mean(a["decode"])), fmt(stdev(a["decode"]))))
        L.append("")
        L.append("Every token pulls its ten routed experts per layer from the SSD, so the blob one expert costs is the "
                 "number a token actually pays for; the pack file is mostly experts a token never asks for.  Two packs "
                 "whose blobs are the same size land at nearly the same rate, which is why the Unsloth 4.5-bit set is "
                 "slower than the 2-bit one even though their files differ by less than 2x.")
        L.append("")
    L.append("")
    trows = [(c[1], types_text(c[1])) for c in CONFIGS if agg.get(c[1]) and agg[c[1]]["n_decode"]]
    trows = [x for x in trows if x[1] != "—"]
    if trows:
        L.append("## What the packs are made of")
        L.append("")
        L.append("| Pack | Gate/up / down experts (layers) | Largest blob | Decode, steady |")
        L.append("| --- | --- | ---: | ---: |")
        for label, txt in trows:
            b = blob_mb(label)
            a = agg[label]
            L.append("| `%s` | %s | %s MB | %s tok/s |" % (label, txt, ("%.2f" % b) if b else "—",
                                                            fmt(mean(a["decode"]))))
        L.append("")
        L.append("The formats are the packs' own metadata (`native_experts.txt`), and they are what the blob column "
                 "is made of: the ISTA sets spend the same 2–3 bits on every expert of every layer, while the Unsloth "
                 "UD sets mix formats layer by layer to keep the important ones wider.")
        L.append("")
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
    for n, r in enumerate(sorted(records, key=lambda r: r.get("order", 0)), start=1):
        if r.get("rc") == 0 and "decode_s" in r:
            gen = r.get("generated") or []
            ok = (gen[:len(EXPECT)] == EXPECT) if gen else bool(r.get("answer_ok", False))
            note = ("first round (cold)" if r.get("round", 0) == 0 else "later round") if ok else "answer diverges"

            L.append(f"| {n} | {r.get('round', 0)+1} | `{r['label']}` | {fmt(r.get('prefill_s', 0))} | {fmt(r['decode_s'])} | {fmt(rate(r, 'decode'))} | {note} |")
        else:
            L.append(f"| {n} | {r.get('round', 0)+1} | `{r.get('label','?')}` | — | — | — | failed: {r.get('error', 'unknown')} |")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```sh")
    L.append("# the whole matrix (this is what wrote the page): 5 shuffled rounds, one commit per run")
    L.append("python3 bench/macpro51/run_bench_qwen38.py --rounds 5 --seed 20261118")
    L.append("")
    L.append("# one pack, one run, the way the harness calls the engine")
    L.append("STRATA_METAL_STREAM=1 ./build-bench/strata --pack ~/models/qwen38/Q2_0/pack \\")
    L.append("    --native ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \\")
    L.append("    --ple-gguf ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \\")
    L.append("    --tokens " + ",".join(str(i) for i in IDS[:PREFILL]) + " \\")
    L.append(f"    --max-new {DECODE} --max-context 512 --spec 1 --suffix-draft 0 --lookup-max-t 0 --prefill {PREFILL}")
    L.append("```")
    L.append("")
    L.append("A multi-shard pack (the Unsloth sets) is joined first by `metal/tools/metal_repack.py` and packed with "
             "`tools/iq_pack.py --compat-bf16`; `bench/macpro51/prep_unsloth.sh <PACK>` does both and a smoke run. "
             "The repacked file is passed as `--native` **and** `--native-dense-gguf`.")
    L.append("")
    L.append("## Limits recorded here")
    L.append("")
    L.append("- The pack's second shard is the 28.8 GB PLE n-gram table. It is read from the SSD a few rows at a time and is never loaded whole; its size counts in the download, not in VRAM.")
    L.append("- Metal 2 on this card has no device addresses; kernels take buffer + offset, and the engine hands every weight that way. No Metal 3 feature is used.")
    L.append("- The output head is 512 wide in the draft path and 248 320 wide in the sample; greedy decoding is used throughout so the comparison is not a sampler's.")
    L.append("- Refused, with the reason on each row: `IQ2_XS` needs an IQ1_M expert kernel this branch does not have, "
             "and the Unsloth v3.1 sets (`UD-Q2_0`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-Q2_K_XL`) answer differently from "
             "every working pack - the defect is written up in [qwen38-metal-v3.1-refusal.md](qwen38-metal-v3.1-refusal.md), "
             "localised to layer 1's fold.")
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
    ap.add_argument("--render-only", action="store_true", help="rewrite the page and charts from the results file")
    ap.add_argument("--prefill-only", action="store_true",
                    help="measure the prompt only (one token after it); for a forward that does not answer yet")
    a = ap.parse_args()
    global PREFILL_ONLY
    PREFILL_ONLY = a.prefill_only

    rng = random.Random(a.seed)
    listed = [c for c in CONFIGS if a.only is None or c[1] in a.only]
    if a.list:
        refused = {n for n, _ in REFUSED}
        for c in listed:
            if c[1] in refused:
                print(c[1], "refused")
                continue
            st = "ready" if ready(c[1]) else ("repacked, no pack" if repacked(c[1]) else
                                               ("downloaded" if shards(c[1])[0] else "missing"))
            print(c[1], st)
        return 0
    # a pack that is not ready, or that this branch refuses, is not a run: it would only add failed rows
    refused = {name for name, _ in REFUSED}
    active = [c for c in listed if ready(c[1]) and c[1] not in refused]
    if not active:
        print("no pack on disk: nothing to run")
        return 0

    records = load_records()
    if a.render_only:
        agg = aggregate(records)
        render_charts(agg)
        render_timeline(records, agg)
        render_doc(agg, records, json.loads((HERE / "meta-qwen38.json").read_text()))
        print("page and charts rewritten from", len(records), "records")
        return 0
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
