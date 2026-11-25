#!/usr/bin/env python3
"""The full suite: prompt/context scaling, thread interaction, and edge cases, on macpro51.

    python3 terminal/bench/macpro51/run_suite.py [--only BLOCK] [--runs N]

Blocks
  scale    prompt sizes in powers of two (8 .. 2048) on UD-IQ1_M, 2 runs each (1 for the largest)
  threads  --pool-workers x prompt size, to show whether the two interact
  edge     the cases a reader actually asks about: 1-token prompt, prompt longer than --max-context,
           --max-new 0, a block size of 2, an invalid block size, --expert-cache 0, --no-pool, an
           unknown flag, and a declared --max-context far above the prompt

Every case is bounded by its own timeout: a hang or an OOM is recorded as the result, not left to
block the suite.  Refusals keep the engine's own line.  `--max-context` is raised with the prompt so
that each scaling point measures prefill rather than a refusal; the model's declared ceiling is
262144 (max_position_embeddings), which is not reachable here in wall-clock terms and is discussed in
the doc rather than attempted.

The breadth block (every model on the box) is `run_all_models.py` and its records are already in
results-all-models.json; this suite does not repeat it.
"""

from __future__ import annotations
import os

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next((q for q in HERE.parents if (q / ".git").exists()), HERE.parents[1])   # the clone root, wherever bench/ lives
DATA = REPO / "terminal" / "data"   # data/ moved under terminal/ in the 2026-11-25 merge
BUILD = Path(os.environ.get("STRATA_BENCH_BUILD", REPO / "build-merge"))
BIN = BUILD / "strata"
M = Path.home() / "models" / "qwen38" / "UD-IQ1_M"
DRAFTER = Path.home() / "models" / "dflash2-drafter" / "dflash2.pack"
OUT = HERE / "results-suite.json"

BASE = [3, 1, 42498, 456, 11, 1544, 76, 398, 8709, 96302, 17776, 78, 76, 88, 13535, 639, 1320]

PREFILL = re.compile(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)
DECODE = re.compile(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)
OUTPUT = re.compile(r"^output\s*: (.*)$", re.M)
SPEC = re.compile(r"speculation\s+\d+ rounds of \d+, drafts accepted (\d+) of (\d+)", re.M)
CPU = re.compile(r"on the CPU (\d+) \(share ([\d.]+)\)")


def tokens_for(n):
    out = []
    while len(out) < n:
        out.extend(BASE)
    return out[:n]


def base_argv(prompt_tokens, max_context, max_new, spec, workers, extra):
    return [str(BIN),
            "--pack", str(M / "pack"),
            "--native", str(M / "UD-IQ1_M-metal.gguf"),
            "--native-dense-gguf", str(M / "UD-IQ1_M-metal.gguf"),
            "--ple-gguf", str(M / "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"),
            "--expert-profile", str(DATA / "expert-profile.bin"), "--expert-cache", "auto",
            "--tokens", ",".join(str(t) for t in tokens_for(prompt_tokens)),
            "--max-new", str(max_new), "--max-context", str(max_context),
            "--draft", "dflash2", "--draft-model", str(DRAFTER),
            "--draft-fallback", "none", "--mtp-draft-vocab", str(DATA / "draft_vocab.bin"),
            "--spec", str(spec), "--suffix-draft", "0", "--lookup-max-t", "0", "--prefill", "1",
            "--pool-workers", str(workers)] + extra


def run_case(case):
    argv = case.pop("argv")
    timeout = case.pop("timeout", 900)
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", "timeout after %d s" % timeout
    rec = dict(case)
    rec["rc"] = rc
    rec["wall_s"] = round(time.time() - t0, 1)
    m = PREFILL.search(out)
    if m:
        rec.update(prefill_tokens=int(m.group(1)), prefill_ms=float(m.group(2)), prefill_tok_s=float(m.group(3)))
    d = DECODE.search(out)
    if d:
        rec.update(decode_tokens=int(d.group(1)), decode_ms=float(d.group(2)), decode_tok_s=float(d.group(3)))
    s = SPEC.search(out)
    if s:
        rec.update(accepted=int(s.group(1)), offered=int(s.group(2)))
    o = OUTPUT.search(out)
    if o:
        rec["tokens_out"] = o.group(1).split()
    c = CPU.search(out)
    if c:
        rec.update(cpu_experts=int(c.group(1)), cpu_share=float(c.group(2)))
    if rc != 0 or "prefill_tok_s" not in rec:
        rec["verdict"] = "refused" if rc not in (0, -9) else ("timeout" if rc == -9 else "ran")
        # the engine prints statistics lines after the real complaint ("shared-event waits ..."), so pick the
        # first line that reads like a diagnosis rather than the last line that happens to contain a keyword
        fatal = ("unknown argument", "must fit", "not built", "cannot read", "is out of range", "holds 2..8",
                 "requires", "no such", "not supported", "cannot open", "does not fit", "refus")
        lines = [ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip()]
        msg = next((ln for ln in lines if any(k in ln.lower() for k in fatal)), None)
        if msg is None:
            msg = next((ln for ln in reversed(lines) if "shared-event waits" not in ln and "wired (mlock)" not in ln),
                       "rc=%d" % rc)
        rec["message"] = msg[:300]
    else:
        rec["verdict"] = "ran"
    return rec


def scale_cases(runs):
    cases = []
    for n in (8, 32, 128, 512, 2048):
        reps = 1 if n >= 2048 else runs
        for r in range(1, reps + 1):
            cases.append({"block": "scale", "case": "prompt %d" % n, "prompt_tokens": n, "round": r,
                          "max_context": max(512, n + 16), "max_new": 2, "spec": 8, "workers": 0,
                          "timeout": 1800,
                          "argv": base_argv(n, max(512, n + 16), 2, 8, 0, [])})
    return cases


def thread_cases():
    cases = []
    for w in (1, 4, 8, 12):
        for n in (32, 256):
            cases.append({"block": "threads", "case": "workers %d, prompt %d" % (w, n), "prompt_tokens": n,
                          "round": 1, "max_context": max(512, n + 16), "max_new": 2, "spec": 8, "workers": w,
                          "timeout": 900,
                          "argv": base_argv(n, max(512, n + 16), 2, 8, w, [])})
    return cases


def edge_cases():
    ok = lambda **kw: {  # noqa: E731  a normal run with overrides
        "argv": base_argv(kw.pop("prompt", 5), kw.pop("ctx", 512), kw.pop("max_new", 2),
                          kw.pop("spec", 8), kw.pop("workers", 0), kw.pop("extra", [])),
        "timeout": kw.pop("timeout", 900), **kw}
    return [
        dict({"block": "edge", "case": "1-token prompt", "prompt_tokens": 1, "round": 1}, **ok(prompt=1, ctx=64, max_new=2)),
        dict({"block": "edge", "case": "prompt 64 with max-context 16", "prompt_tokens": 64, "round": 1}, **ok(prompt=64, ctx=16, max_new=2)),
        dict({"block": "edge", "case": "max-new 0", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, max_new=0)),
        dict({"block": "edge", "case": "verify block 2", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, spec=2)),
        dict({"block": "edge", "case": "verify block 1 (invalid)", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, spec=1)),
        dict({"block": "edge", "case": "verify block 64 (above MAXT?)", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, spec=64)),
        dict({"block": "edge", "case": "no GPU expert cache", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, extra=["--expert-cache", "0"])),
        dict({"block": "edge", "case": "no CPU pool", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, extra=["--no-pool"])),
        dict({"block": "edge", "case": "unknown flag", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, extra=["--definitely-not-a-flag"])),
        dict({"block": "edge", "case": "pool-workers 64 (oversubscribed)", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=512, workers=64)),
        dict({"block": "edge", "case": "max-context 4096, prompt 5 (allocation)", "prompt_tokens": 5, "round": 1}, **ok(prompt=5, ctx=4096, timeout=600)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, help="scale | threads | edge")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    want = set(args.only) if args.only else {"scale", "threads", "edge"}
    cases = []
    if "scale" in want:
        cases += scale_cases(args.runs)
    if "threads" in want:
        cases += thread_cases()
    if "edge" in want:
        cases += edge_cases()

    recs = json.loads(Path(args.out).read_text()) if Path(args.out).is_file() else []
    done = {(r["block"], r["case"], r["round"]) for r in recs}
    for c in cases:
        key = (c["block"], c["case"], c["round"])
        if key in done:
            continue
        rec = run_case(dict(c))
        recs.append(rec)
        Path(args.out).write_text(json.dumps(recs, indent=1) + "\n")
        print("%-8s %-42s %-8s %s" % (rec["block"], rec["case"], rec["verdict"],
                                      ("prefill %.1f ms (%.2f tok/s) wall %.1f s" % (rec.get("prefill_ms", 0), rec.get("prefill_tok_s", 0), rec["wall_s"]))
                                      if rec["verdict"] == "ran" else rec.get("message", "")[:80]), flush=True)
    print("wrote %s (%d records)" % (args.out, len(recs)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
