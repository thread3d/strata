#!/usr/bin/env python3
"""How far apart two packs' logits are, on one prompt, through the engine's own --dump-logits.

Token identity is too coarse to steer a kernel fix: two packs can produce the same first
tokens and still sit a whole logit apart. This prints the per-position max and RMS distance
in logits plus whether the argmax agrees, so a change to the down kernel can be judged
quantitatively instead of by "the tokens look right now".

    python3 terminal/bench/macpro51/logit_distance.py --ref Q8_0 --test Q6_K

Row 0 is the first prompt token's row. It is reported but excluded from the summary: the
engine's first row is not a prediction the model ever makes, and on this machine it comes out
with a margin of 1e24, so including it would swamp the averages.
"""

from __future__ import annotations

import argparse
import array
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "build-metal" / "strata-q36-metal"
MODELS = Path.home() / "models" / "qwen36"

CONTEXT = "3,1,42498,456,11,303,1330,2716,22157,11,3069,279,12515,5686,6105,2261,279,1834,321,2438,506,41564,13"


def dump(pack: str, path: Path, vocab: int) -> int:
    model = MODELS / ("Qwen3.6-35B-A3B-%s.gguf" % pack)
    if not model.is_file():
        print("no such pack: %s" % model)
        return 1
    proc = subprocess.run(
        [str(BIN), "--model", str(model), "--tokens", CONTEXT, "--n", "1",
         "--dump-logits", str(path)],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or not path.is_file():
        print("dump failed for %s: %s" % (pack, (proc.stderr or proc.stdout).strip().splitlines()[-1:]))
        return 1
    return 0


def load(path: Path) -> array.array:
    a = array.array("f")
    with path.open("rb") as fh:
        a.fromfile(fh, os.path.getsize(path) // 4)
    return a


def compare(ref, test, vocab: int, keep_row0: bool):
    rows = len(ref) // vocab
    out = []
    for r in range(1 if not keep_row0 else 0, rows):
        o = r * vocab
        mx = ss = 0.0
        ai = bi = o
        for i in range(vocab):
            d = ref[o + i] - test[o + i]
            if d < 0.0:
                d = -d
            if d > mx:
                mx = d
            ss += d * d
            if ref[o + i] > ref[ai]:
                ai = o + i
            if test[o + i] > test[bi]:
                bi = o + i
        out.append((r, mx, (ss / vocab) ** 0.5, ai - o, bi - o))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Q8_0")
    ap.add_argument("--test", default="Q6_K")
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--keep-row0", action="store_true")
    args = ap.parse_args()

    if not BIN.is_file():
        print("no engine at %s" % BIN)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="macpro51-logits-"))
    a, b = tmp / ("%s.bin" % args.ref), tmp / ("%s.bin" % args.test)
    if dump(args.ref, a, args.vocab) or dump(args.test, b, args.vocab):
        return 1

    rows = compare(load(a), load(b), args.vocab, args.keep_row0)
    agree = sum(1 for r in rows if r[3] == r[4])
    print("%-10s vs %-10s  (logits, %d positions)" % (args.ref, args.test, len(rows)))
    print("  row   max     rms     argmax %s / %s" % (args.ref, args.test))
    for r, mx, rms, ai, bi in rows:
        print("  %3d  %6.3f  %6.3f   %7d %7d %s" % (r, mx, rms, ai, bi, "" if ai == bi else "<- differs"))
    med = sorted(x[2] for x in rows)[len(rows) // 2]
    print("  median rms %.4f, max rms %.4f, argmax agrees on %d of %d positions"
          % (med, max(x[2] for x in rows), agree, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
