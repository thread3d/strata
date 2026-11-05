#!/bin/bash
# bench-suite.sh - run every model-free benchmark on the Jetson, N times each, with one discarded warm-up run, and
# record GPU/CPU clocks and temperatures alongside so a spread can be read.
#
#   N=5 sh bench-suite.sh            # 1 warm-up + 5 measured runs of every tool
#   ONLY="q8_gemv_bench" N=2 sh ...  # one tool, for a quick check
#
# Writes bench/results/jetson-agx-xavier/runs/<tool>/run<NN>.txt, .../manifest.tsv (start/end per run) and
# .../tegrastats.txt for the whole session.
set -u
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)" || exit 1   # the repo root
B=build-jetson
OUT=bench/results/jetson-agx-xavier
RUNS=$OUT/runs
N=${N:-5}                     # measured runs per tool; run 0 is always the discarded warm-up
ONLY=${ONLY:-}
mkdir -p "$RUNS"

# tool name -> arguments.  One line per tool; keep the arguments with the tool so a rerun matches.
matrix() {
cat <<'EOF'
dflash2_bw_bench|
dflash2_launch_bench|100
dflash2_gemm_bench|
dflash2_small_bench|
dflash2_attn_bench|
router_chain_bench_T1|1 20
router_chain_bench_T2|2 20
router_chain_bench_T4|4 20
tq_pascal_bench|--bench
qsa_rows_bench|
q8_gemv_bench|--iters 50
gdn_chunk_bench|
qsa_prompt_attn_sm70_bench|32768 2048 5
mmq_expert_bench|4 16
nvfp4_decode_bench|
nvfp4_decode_bench_iq|21 20 64 2 50
b6_mmvq_bench|200 -1 1
expdec_bench|50 -1 20
EOF
}

# tegrastats for the whole session, timestamped
tegrastats --interval 500 2>/dev/null | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$OUT/tegrastats.txt" &
TS_PID=$!
trap 'kill $TS_PID 2>/dev/null' EXIT INT TERM

: > "$OUT/manifest.tsv"
printf 'tool\trun\tkind\targs\tstart\tend\texit\n' >> "$OUT/manifest.tsv"

total=0; done_n=0
matrix | while IFS='|' read -r name args; do total=$((total+1)); echo "$total $name"; done > /tmp/toolcount
TOTAL=$(tail -1 /tmp/toolcount | cut -d' ' -f1)
echo "tools: $TOTAL   measured runs each: $N   (+1 warm-up)"

i=0
matrix | while IFS='|' read -r name args; do
  i=$((i+1))
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  bin=$B/$name
  # per-tool variants share one binary
  case "$name" in router_chain_bench_*) bin=$B/router_chain_bench ;; nvfp4_decode_bench_iq) bin=$B/nvfp4_decode_bench ;; esac
  if [ ! -x "$bin" ]; then echo "[$i/$TOTAL] $name: NOT BUILT"; continue; fi
  mkdir -p "$RUNS/$name"
  for r in $(seq 0 "$N"); do
    kind=measured; [ "$r" = 0 ] && kind=warmup
    f=$(printf '%s/%s/run%02d.txt' "$RUNS" "$name" "$r")
    t0=$(date +%s.%N)
    timeout 1800 "$bin" $args > "$f" 2>&1
    code=$?
    t1=$(date +%s.%N)
    printf '%s\t%s\t%s\t%s\t%.3f\t%.3f\t%s\n' "$name" "$r" "$kind" "${args:-none}" "$t0" "$t1" "$code" >> "$OUT/manifest.tsv"
    printf '  %-26s %-8s exit=%-3s %6.1fs\n' "$name" "$kind" "$code" "$(echo "$t1 - $t0" | bc)"
  done
done
echo "done: $OUT"
