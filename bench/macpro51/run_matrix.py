#!/usr/bin/env python3
"""The Qwen3.6-35B-A3B matrix on macpro51, one engine process per run.

    python3 bench/macpro51/run_matrix.py --runs 2
    python3 bench/macpro51/run_matrix.py --only Q8_0 --runs 1

Writes ``results-qwen36.json`` next to this file.  The engine prints everything this script
reads; nothing is inferred.  Packs the engine refuses are recorded with the engine's own line
(they are never timed).  ``charts.py`` turns the JSON into the CSV and the SVG beside it.
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
BIN = REPO / "build-metal" / "strata-q36-metal"
MODELS = Path.home() / "models" / "qwen36"
RESULTS = HERE / "results-qwen36.json"

PROMPT = ("Explain, in two short sentences, why the sky looks blue during the day "
          "and red at sunset.")
DECODE = 24

# Lightest first.  The two at the end are the packs the engine refuses; they are here so that a
# refusal is recorded as a result instead of quietly dropping out of the matrix.
PACKS = ["Q4_K_M", "UD-Q4_K_XL", "Q6_K", "UD-Q6_K_XL", "Q8_0", "UD-Q2_K_XL", "UD-Q8_K_XL"]

STEP = re.compile(r"prompt (\d+) token\(s\) \+ (\d+) generated in ([\d.]+) s \(([\d.]+) s a step\)")
TOKENS = re.compile(r"^tokens: (.*)$", re.M)
SLAB = re.compile(r"slab: ([\d.]+) MiB \(([^)]*)\)")


def run_one(pack: str, round_no: int) -> dict:
    path = MODELS / ("Qwen3.6-35B-A3B-%s.gguf" % pack)
    rec = {"pack": pack, "round": round_no, "model": str(path)}
    if not path.is_file():
        rec.update(rc=-1, error="no such file")
        return rec
    rec["bytes"] = path.stat().st_size
    started = time.time()
    proc = subprocess.run(
        [str(BIN), "--model", str(path), "--prompt", PROMPT, "--n", str(DECODE), "--text"],
        capture_output=True, text=True, timeout=5400,
    )
    rec["wall_s"] = round(time.time() - started, 1)
    rec["rc"] = proc.returncode
    out = proc.stdout
    step = STEP.search(out)
    if step:
        rec["prompt_tokens"] = int(step.group(1))
        rec["decode_tokens"] = int(step.group(2))
        rec["seconds"] = float(step.group(3))
        rec["s_per_step"] = float(step.group(4))
    tok = TOKENS.search(out)
    if tok:
        rec["tokens"] = [int(t) for t in tok.group(1).split()]
    slab = SLAB.search(out)
    if slab:
        rec["slab_mib"] = float(slab.group(1))
        rec["slab"] = slab.group(2)
    if not step:
        # A refused pack names the tensor it cannot take; keep that line, not a guess at it.
        for line in (out + proc.stderr).splitlines():
            if "cannot read the model" in line or "error" in line.lower():
                rec["error"] = line.strip()
                break
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--only", action="append", default=None, help="a pack name; repeatable")
    args = ap.parse_args()

    packs = args.only or PACKS
    if not BIN.is_file():
        print("no engine at %s; build it first" % BIN)
        return 2

    records = []
    if RESULTS.is_file():
        records = json.loads(RESULTS.read_text())
    done = {(r["pack"], r["round"]) for r in records}
    for round_no in range(1, args.runs + 1):
        for pack in packs:
            if (pack, round_no) in done:
                continue
            rec = run_one(pack, round_no)
            records.append(rec)
            RESULTS.write_text(json.dumps(records, indent=1) + "\n")
            if "s_per_step" in rec:
                print("%-12s run %d  %.3f s/step  %.1f s wall" % (
                    pack, round_no, rec["s_per_step"], rec["wall_s"]))
            else:
                print("%-12s run %d  refused: %s" % (pack, round_no, rec.get("error", "rc=%s" % rec["rc"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
