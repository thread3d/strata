#!/bin/bash
# Wait until every pack that can be prepared is, then pause the downloads, run the full mixed sweep, and restart
# them (the trap guarantees the restart even if the sweep dies).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="${LOG:-$HOME/logs/final-sweep.log}"
cd "$HERE/../.." || exit 1
stamp() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> "$LOG"; }
start_dl() { screen -dmS macpro51-qwen38-dl3 bash -c 'bash "$HERE/download_qwen38.sh" > "$HOME/logs/qwen38-dl3.log" 2>&1'; }
start_prep() { screen -dmS macpro51-prep-all bash -c 'bash "$HERE/prep_all_unsloth.sh" > "$HOME/logs/prep-all-unsloth.log" 2>&1'; }
finish() { start_dl; start_prep; stamp "downloads and prep restarted"; }
trap finish EXIT
stamp "waiting for ALL PREPARED"
for i in $(seq 1 480); do
  grep -q "ALL PREPARED" "$HOME/logs/prep-all-unsloth.log" 2>/dev/null && break
  sleep 60
done
stamp "pausing the downloads"
screen -S macpro51-qwen38-dl3 -X quit 2>/dev/null
screen -S macpro51-prep-all -X quit 2>/dev/null
pkill -f qwen38_download_unsloth.sh 2>/dev/null
pkill -f prep_all_unsloth.sh 2>/dev/null
pkill -f "curl.*Qwen3.8-Flash-Next" 2>/dev/null
pkill -f metal_repack 2>/dev/null
sleep 5
git mv bench/macpro51/results-qwen38.jsonl bench/macpro51/results-qwen38-7pack.jsonl 2>/dev/null
git add bench/macpro51/results-qwen38-7pack.jsonl 2>/dev/null
git commit -q -m "[Benchmark] macpro51 qwen38: keep the seven-pack matrix as a record before the full sweep" 2>/dev/null
stamp "sweep start"
python3 bench/macpro51/run_bench_qwen38.py --rounds 5 --seed 20261118 >> "$LOG" 2>&1
stamp "SWEEP_EXIT=$?"
