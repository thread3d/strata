#!/usr/bin/env python3
"""Sweep the CPU expert-pool worker count and record what the prompt path does with it.

    python3 bench/macpro51/run_thread_sweep.py --runs 3

One engine process per (workers, run).  The workload is a fixed 260-token prompt with a 2-token decode:
the prompt is where the CPU pool works here -- `--pool-workers` sizes the CPU expert pool, and the default
prefill path gives the idle CPU a share of the experts the GPU would otherwise stream.  The decode window
on this model is GPU-bound and reports "CPU experts 0.00 routed per layer", so decode is recorded but is
not expected to move.

Each record keeps the engine's own lines (prefill tokens/ms, decode ms, the token ids) so nothing has to be
re-derived; the sweep appends to the JSON as it goes, so an interrupted run resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "build-metal" / "strata"
MODELS = Path.home() / "models" / "qwen38" / "UD-IQ1_M"
RESULTS = HERE / "results-threads.json"

# 260 tokens: the model's own continuation repeated, so the prompt is long enough for the CPU share to run
PROMPT = ",".join(["3", "1", "42498", "456", "11", "1544", "76", "398", "8709", "96302", "17776", "78", "76"] * 20)

PREFILL = re.compile(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)
DECODE = re.compile(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)
OUTPUT = re.compile(r"^output\s*: (.*)$", re.M)
WORKERS = re.compile(r"^strata generate: (\d+) pool workers on logical processors.*$", re.M)
# did the CPU pool actually receive work?  "on the CPU 0 (share 0.00)" / "CPU experts 0.00 routed"
CPU_PRE = re.compile(r"on the CPU (\d+) \(share ([\d.]+)\)")
CPU_DEC = re.compile(r"CPU experts ([\d.]+) distinct / ([\d.]+) routed per layer")


def run(workers: int, round_no: int) -> dict:
    argv = [str(BIN),
            "--pack", str(MODELS / "pack"),
            "--native", str(MODELS / "UD-IQ1_M-metal.gguf"),
            "--native-dense-gguf", str(MODELS / "UD-IQ1_M-metal.gguf"),
            "--ple-gguf", str(MODELS / "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"),
            "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
            "--tokens", PROMPT, "--max-new", "2", "--max-context", "512",
            "--draft", "dflash2", "--draft-model", str(Path.home() / "models/dflash2-drafter/dflash2.pack"),
            "--draft-fallback", "none", "--mtp-draft-vocab", "data/draft_vocab.bin",
            "--spec", "8", "--suffix-draft", "0", "--lookup-max-t", "0", "--prefill", "1",
            "--pool-workers", str(workers)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    rec = {"workers": workers, "round": round_no, "rc": p.returncode, "wall_s": round(wall, 1)}
    m = PREFILL.search(p.stdout)
    if m:
        rec.update(prefill_tokens=int(m.group(1)), prefill_ms=float(m.group(2)), prefill_tok_s=float(m.group(3)))
    d = DECODE.search(p.stdout)
    if d:
        rec.update(decode_tokens=int(d.group(1)), decode_ms=float(d.group(2)), decode_tok_s=float(d.group(3)))
    o = OUTPUT.search(p.stdout)
    if o:
        rec["tokens"] = o.group(1).split()
    w = WORKERS.search(p.stderr)
    if w:
        rec["reported_workers"] = int(w.group(1))
    cp = CPU_PRE.search(p.stdout + p.stderr)
    if cp:
        rec["cpu_prefill_experts"] = int(cp.group(1))
        rec["cpu_prefill_share"] = float(cp.group(2))
    cd = CPU_DEC.search(p.stdout + p.stderr)
    if cd:
        rec["cpu_decode_distinct"] = float(cd.group(1))
        rec["cpu_decode_routed"] = float(cd.group(2))
    if p.returncode != 0:
        rec["error"] = (p.stderr.strip().splitlines() or ["rc=%d" % p.returncode])[-1]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--points", default="1,2,3,4,5,6,7,8,9,10,11,12,24",
                    help="pool worker counts; 12 = the physical cores, 24 = the logical maximum (oversubscribed)")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    points = [int(x) for x in args.points.split(",")]
    records = json.loads(Path(args.out).read_text()) if Path(args.out).is_file() else []
    done = {(r["workers"], r["round"]) for r in records}
    for round_no in range(1, args.runs + 1):
        for w in points:
            if (w, round_no) in done:
                continue
            rec = run(w, round_no)
            records.append(rec)
            Path(args.out).write_text(json.dumps(records, indent=1) + "\n")
            print("workers %2d  run %d  prefill %8.1f ms (%.2f tok/s)  decode %7.1f ms  wall %5.1f s  rc=%d"
                  % (w, round_no, rec.get("prefill_ms", float("nan")), rec.get("prefill_tok_s", float("nan")),
                     rec.get("decode_ms", float("nan")), rec["wall_s"], rec["rc"]), flush=True)
    print("wrote %s (%d records)" % (args.out, len(records)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
