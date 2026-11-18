#!/bin/bash
# Repack one downloaded Unsloth pack into the single GGUF the Metal window opens, build its pack metadata,
# and smoke-test it. Usage: prep_unsloth.sh <PACK>
set -u
PACK=${1:?usage: prep_unsloth.sh PACK}
D="$HOME/models/qwen38/$PACK"
cd "$HOME/strata-bench"
export STRATA_GGUF_PY="$HOME/projects/llama.cpp/gguf-py"   # iq_pack --compat-bf16 needs gguf-py
S1=$(ls "$D"/*-00001-of-*.gguf 2>/dev/null | head -1)
if [ -z "$S1" ]; then
  # a single-file pack (Gyro: one GGUF, the n-gram table inside): no repack, the file is the native source and
  # the PLE source both
  S1=$(ls "$D"/*.gguf 2>/dev/null | head -1)
  [ -n "$S1" ] || { echo "no GGUF in $D"; exit 1; }
  OUT="$S1"
  echo "single-file pack: $OUT (no repack)"
else
OUT="$D/$PACK-metal.gguf"
fi
if [ ! -f "$OUT" ]; then
  echo "== repack $(date -u) =="
  /usr/bin/time -p python3 metal/tools/metal_repack.py --shard1 "$S1" --out "$OUT" --verify 2>&1 | tee "$HOME/logs/repack-$PACK.log" | tail -6
else
  echo "repacked file already there: $OUT"
fi
PLE=$(grep -a "n-gram table:" "$HOME/logs/repack-$PACK.log" 2>/dev/null | tail -1 | sed 's/^n-gram table: //')
[ -n "$PLE" ] || PLE=$(python3 - "$D" <<'PY'
import sys, glob, struct, os
d=sys.argv[1]
for p in sorted(glob.glob(d+"/*-of-*.gguf")):
    b=open(p,'rb').read(24<<20)
    if b"per_layer_token_embd.weight" in b:
        print(p); break
PY
)
[ -n "$PLE" ] || PLE="$S1"   # a single-file pack keeps the n-gram table inside itself
echo "PLE shard: $PLE"
echo "== iq_pack $(date -u) =="
python3 tools/iq_pack.py --gguf "$OUT" --out "$D/pack" --compat-bf16 2>&1 | tail -3
echo "== smoke $(date -u) =="
IDS="814,20139,11,303,1330,2716,22157,11,3069,279,12515,5686,6105,2261,279,1834,321,2438,506,41564,13,4844,1910,3069,279,16820,5686,18016,3043,279,32946,13"
STRATA_METAL_STREAM=1 ./build-bench/strata --pack "$D/pack" --native "$OUT" --ple-gguf "$PLE" --native-dense-gguf "$OUT" \
  --tokens "$IDS" --max-new 8 --max-context 512 --spec 1 --suffix-draft 0 --lookup-max-t 0 --prefill 32 2>&1 | \
  grep -E "output  :|decode  |prefill |crashed|signal|error|no function" | head -6
echo "== done $(date -u) =="
