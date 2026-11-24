#!/usr/bin/env python3
"""Exercise every model on this machine and record what the engine says, one process per model.

    python3 bench/macpro51/run_all_models.py [--only NAME] [--out results-all-models.json]

Three families are on the box and each is driven by a different entry point:

  * qwen36  - plain GGUF, `build-metal/strata-q36-metal --model X --prompt "..." --n N --text`
              (the q36 engine reads one file; the seven packs include two the engine refuses).
  * qwen38  - native pack + a metal GGUF + the PLE shard, `build-metal/strata --pack ... --native ...
              --ple-gguf ... --draft dflash2 ...`.  A native IQ pack refuses without `--spec`, and
              `--spec` needs a draft source, so the DFlash2 drafter is always attached here.
  * deepseek-v4.1-flash - `strata_ds4 --pack-info` (works without a GPU); running it needs
              STRATA_DS4_ALLOW_UNSTABLE_METAL=1, which resets this card, so it is recorded, not run.

Refusals are results: the engine's own line is kept verbatim rather than summarised.  Nothing here is
timed for performance - the numbers are only there to say whether a model runs at all and how fast it
gets through a fixed short workload on this card.
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
BIN_Q36 = REPO / "build-metal" / "strata-q36-metal"
BIN_DS4 = REPO / "build-metal" / "strata_ds4"
MODELS = Path.home() / "models"
RESULTS = HERE / "results-all-models.json"

PROMPT = "Explain, in two short sentences, why the sky looks blue during the day and red at sunset."
Q38_TOKENS = "3,1,42498,456,11"
PLE_BYTES = 28800138432   # the PLE/vision shard has this size in every qwen38 pack

STEP = re.compile(r"prompt (\d+) token\(s\) \+ (\d+) generated in ([\d.]+) s \(([\d.]+) s a step\)")
SPEC = re.compile(r"speculation\s+\d+ rounds of \d+, drafts accepted (\d+) of (\d+)")
OUTPUT = re.compile(r"^output\s*: (.*)$", re.M)
PREFILL = re.compile(r"^prefill\s+(\d+) tokens in ([\d.]+) ms", re.M)


def rc(name, argv, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        return -9, "", "timeout after %d s" % timeout, round(time.time() - t0, 1)


def first_error(text):
    for line in text.splitlines():
        low = line.lower()
        if "cannot read the model" in low or "refus" in low or "error" in low or "not built" in low:
            return line.strip()
    return None


def q36(name):
    path = MODELS / "qwen36" / name
    rec = {"family": "qwen36", "name": name, "model": str(path), "bytes": path.stat().st_size if path.is_file() else 0}
    code, out, err, wall = rc(name, [str(BIN_Q36), "--model", str(path), "--prompt", PROMPT, "--n", "8", "--text"], 3600)
    rec.update(rc=code, wall_s=wall)
    m = STEP.search(out)
    if m:
        rec.update(prompt_tokens=int(m.group(1)), decode_tokens=int(m.group(2)), s_per_step=float(m.group(4)))
    o = OUTPUT.search(out)
    if o:
        rec["tokens"] = o.group(1).split()
    if "s_per_step" not in rec:
        rec["error"] = first_error(out + err) or ("rc=%s" % code)
    return rec


def q38(dirname):
    d = MODELS / "qwen38" / dirname
    rec = {"family": "qwen38", "name": dirname, "dir": str(d)}
    metal = sorted(d.glob("*-metal.gguf"))
    pack = d / "pack"
    ple = [f for f in sorted(d.glob("*.gguf")) if f.stat().st_size == PLE_BYTES]
    if not metal or not pack.is_dir():
        rec.update(rc=-1, error="no native pack or no *-metal.gguf")
        return rec
    if not ple:
        rec.update(rc=-1, error="no PLE shard (none of %d bytes)" % PLE_BYTES)
        return rec
    rec["model"] = str(metal[0])
    rec["bytes"] = metal[0].stat().st_size
    argv = [str(BIN), "--pack", str(pack), "--native", str(metal[0]), "--native-dense-gguf", str(metal[0]),
            "--ple-gguf", str(ple[0]), "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
            "--tokens", Q38_TOKENS, "--max-new", "2", "--max-context", "512",
            "--draft", "dflash2", "--draft-model", str(MODELS / "dflash2-drafter" / "dflash2.pack"),
            "--draft-fallback", "none", "--mtp-draft-vocab", "data/draft_vocab.bin",
            "--spec", "8", "--suffix-draft", "0", "--lookup-max-t", "0", "--prefill", "1"]
    code, out, err, wall = rc(dirname, argv, 5400)
    rec.update(rc=code, wall_s=wall)
    s = SPEC.search(out)
    if s:
        rec.update(accepted=int(s.group(1)), offered=int(s.group(2)))
    o = OUTPUT.search(out)
    if o:
        rec["tokens"] = o.group(1).split()
    p = PREFILL.search(out)
    if p:
        rec.update(prefill_tokens=int(p.group(1)), prefill_ms=float(p.group(2)))
    if "tokens" not in rec:
        rec["error"] = first_error(out + err) or ("rc=%s" % code)
    return rec


def ds4():
    rec = {"family": "deepseek-v4.1-flash", "name": "deepseek-v4.1-flash"}
    code, out, err, wall = rc("ds4", [str(BIN_DS4), "--pack-info"], 600)
    rec.update(rc=code, wall_s=wall)
    for line in (out + err).splitlines():
        if "pack" in line.lower() and ":" in line:
            rec["info"] = line.strip()
            break
    rec["note"] = ("not run: it needs STRATA_DS4_ALLOW_UNSTABLE_METAL=1, which resets this card "
                   "(see AGENTS.md); --pack-info is the part that runs here")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    records = []
    for path in sorted((MODELS / "qwen36").glob("*.gguf")):
        if args.only and path.name not in args.only and path.stem not in args.only:
            continue
        print("qwen36 %-40s" % path.name, flush=True)
        records.append(q36(path.name))
        print("   -> %s" % (records[-1].get("error") or "%s s/step, %s tok" % (records[-1].get("s_per_step"), len(records[-1].get("tokens", [])))), flush=True)

    for d in sorted((MODELS / "qwen38").iterdir()):
        if not d.is_dir() or not (d / "pack").is_dir():
            continue
        if args.only and d.name not in args.only:
            continue
        print("qwen38 %-40s" % d.name, flush=True)
        records.append(q38(d.name))
        print("   -> %s" % (records[-1].get("error") or "acc %s/%s, %s tok, prefill %s ms" % (
            records[-1].get("accepted"), records[-1].get("offered"), len(records[-1].get("tokens", [])),
            records[-1].get("prefill_ms"))), flush=True)

    if (MODELS / "deepseek-v4.1-flash").is_dir() and not args.only:
        print("deepseek-v4.1-flash", flush=True)
        records.append(ds4())

    Path(args.out).write_text(json.dumps(records, indent=1) + "\n")
    print("wrote %s (%d records)" % (args.out, len(records)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
