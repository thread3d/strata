#!/usr/bin/env python3
"""The llama.cpp side of the macpro51 comparison: same packs, same context, same token budget.

    python3 bench/macpro51/run_reference.py --runs 2

The reference is ``llama_ref.cpp`` beside this file, built against the llama.cpp checkout at
``~/projects/llama.cpp`` (rev 8bed7cfb9, CPU + BLAS; there is no Metal backend in that build).  It
takes the context as token ids and prints the greedy continuation as token ids, which is the same
contract as ``strata-q36-metal --tokens`` - so the two engines can be compared id for id instead of
by eye over decoded text.  Build it with:

    cd ~/projects/llama.cpp
    clang++ -O2 -std=c++17 -Iinclude -Iggml/include \
        ~/projects/Strata/strata/bench/macpro51/llama_ref.cpp \
        -Lbuild/bin -lllama -lggml -lggml-base -o ~/llama_ref

and point --ref-bin at it (the default is ~/llama_ref).

Writes ``results-reference.json`` beside this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = Path.home() / "models" / "qwen36"
RESULTS = HERE / "results-reference.json"
DUMP = Path.home() / "llama_ref"
LLAMA_BIN = Path.home() / "projects" / "llama.cpp" / "build" / "bin"

# The 23 ids strata-q36-metal printed for the prompt the matrix uses (see run_matrix.py).  Feeding
# ids rather than text keeps both engines on one tokenisation.
CONTEXT = [3, 1, 42498, 456, 11, 303, 1330, 2716, 22157, 11, 3069, 279, 12515, 5686, 6105,
           2261, 279, 1834, 321, 2438, 506, 41564, 13]
DECODE = 24

PACKS = ["Q4_K_M", "UD-Q4_K_XL", "Q6_K", "UD-Q6_K_XL", "Q8_0"]
GEN = re.compile(r"^gen: (.*)$", re.M)


def run_one(pack: str, round_no: int, ref_bin: Path) -> dict:
    path = MODELS / ("Qwen3.6-35B-A3B-%s.gguf" % pack)
    rec = {"pack": pack, "round": round_no, "model": str(path), "decode": DECODE}
    if not path.is_file():
        rec.update(rc=-1, error="no such file")
        return rec
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = str(LLAMA_BIN)
    started = time.time()
    proc = subprocess.run(
        [str(ref_bin), "--model", str(path), "--tokens", ",".join(str(i) for i in CONTEXT),
         "--n", str(DECODE)],
        capture_output=True, text=True, timeout=5400, env=env,
    )
    rec["wall_s"] = round(time.time() - started, 1)
    rec["rc"] = proc.returncode
    gen = GEN.search(proc.stdout)
    if gen:
        rec["tokens"] = [int(t) for t in gen.group(1).split()]
    else:
        rec["error"] = (proc.stderr.strip().splitlines() or ["no gen line"])[-1]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--ref-bin", default=str(DUMP))
    args = ap.parse_args()

    ref_bin = Path(args.ref_bin)
    if not ref_bin.is_file():
        print("no reference helper at %s (build llama_ref.cpp first)" % ref_bin)
        return 2

    records = json.loads(RESULTS.read_text()) if RESULTS.is_file() else []
    done = {(r["pack"], r["round"]) for r in records}
    for round_no in range(1, args.runs + 1):
        for pack in (args.only or PACKS):
            if (pack, round_no) in done:
                continue
            rec = run_one(pack, round_no, ref_bin)
            records.append(rec)
            RESULTS.write_text(json.dumps(records, indent=1) + "\n")
            if "tokens" in rec:
                print("%-12s run %d  %5.1f s wall  %s" % (
                    pack, round_no, rec["wall_s"], " ".join(map(str, rec["tokens"][:6])) + " ..."))
            else:
                print("%-12s run %d  failed: %s" % (pack, round_no, rec.get("error")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
