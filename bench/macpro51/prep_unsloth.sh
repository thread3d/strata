#!/bin/bash
# bench/macpro51/prep_unsloth.sh - an Unsloth UD-* pack as the one GGUF the Metal window opens, its pack metadata,
# and a smoke run. Run it from the repo root (~/strata-bench on macpro51).
#
#   bench/macpro51/prep_unsloth.sh UD-Q4_K_XL
#
# 1. metal/tools/metal_repack.py joins the 3-6 shards into <PACK>/<PACK>-metal.gguf (the PLE table stays in its shard),
#    which must be passed as --native AND --native-dense-gguf (the window's --native alone would add the PLE shard to
#    the dense list and the shard-split check refuses a repacked file beside it).
# 2. tools/iq_pack.py --compat-bf16 builds the metadata pack (the Unsloth sets keep the HC projections as Q8_0, which
#    the engine's dense reader wants as BF16; the conversion is lossy only in the last bit of those projections).
# 3. A smoke run: 32 prompt ids (the pack's own tokenizer) + 8 greedy tokens.
set -u
PACK=${1:?usage: prep_unsloth.sh PACK}
D="$HOME/models/qwen38/$PACK"
cd "$HOME/strata-bench"
export STRATA_GGUF_PY="$HOME/projects/llama.cpp/gguf-py"   # iq_pack --compat-bf16 needs gguf-py
S1=$(ls "$D"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -n "$S1" ] || { echo "no shard 1 in $D"; exit 1; }
OUT="$D/$PACK-metal.gguf"
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
echo "PLE shard: $PLE"
echo "== iq_pack $(date -u) =="
python3 tools/iq_pack.py --gguf "$OUT" --out "$D/pack" --compat-bf16 2>&1 | tail -3
echo "== smoke $(date -u) =="
IDS="814,20139,11,303,1330,2716,22157,11,3069,279,12515,5686,6105,2261,279,1834,321,2438,506,41564,13,4844,1910,3069,279,16820,5686,18016,3043,279,32946,13"
STRATA_METAL_STREAM=1 ./build-bench/strata --pack "$D/pack" --native "$OUT" --ple-gguf "$PLE" \
  --tokens "$IDS" --max-new 8 --max-context 512 --spec 1 --suffix-draft 0 --lookup-max-t 0 --prefill 32 2>&1 | \
  grep -E "output  :|decode  |prefill |crashed|signal|error|no function" | head -6
echo "== done $(date -u) =="
