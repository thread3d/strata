#!/bin/bash
set -e

MODEL_DIR="$HOME/projects/Strata-data/models"
BIN="$HOME/projects/Strata/strata/build-metal/strata-q36-metal"
TOKENS="760,6511,314"
N_TOKENS=10
OUTPUT_DIR="$HOME/benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

models=(
    "Q8_0:$HOME/models/qwen36/Qwen3.6-35B-A3B-Q8_0.gguf"
    "UD-Q4_K_XL:$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    "UD-Q6_K_XL:$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
)

echo "Benchmarking Qwen3.6 Metal on Radeon RX 580"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

echo "model,run,total_time_s,time_per_token_s,tokens" > "$OUTPUT_DIR/summary.csv"

for model_spec in "${models[@]}"; do
    name="${model_spec%%:*}"
    path="${model_spec#*:}"
    
    if [[ ! -f "$path" ]]; then
        echo "SKIP $name: $path not found"
        continue
    fi
    
    echo "Testing $name..."
    echo "  Path: $path"
    
    for run in 1; do
        echo "  Run $run..."
        output=$("$BIN" --model "$path" --tokens "$TOKENS" --n "$N_TOKENS" 2>&1 | tail -3)
        echo "$output" | tee -a "$OUTPUT_DIR/${name}_run${run}.txt"
        
        time_line=$(echo "$output" | grep "prompt")
        if [[ -n "$time_line" ]]; then
            total_time=$(echo "$time_line" | sed -E "s/.*in ([0-9.]+) s.*/\1/")
            per_token=$(echo "$time_line" | sed -E "s/.*\\(([0-9.]+) s a step\\)).*/\1/")
            tokens=$(echo "$output" | grep "tokens:" | sed "s/.*tokens: //")
            echo "$name,$run,$total_time,$per_token,$tokens" >> "$OUTPUT_DIR/summary.csv"
        fi
    done
    
    echo "  Done $name"
done

echo "============================================"
echo "Benchmark complete. Results in $OUTPUT_DIR"
cat "$OUTPUT_DIR/summary.csv"
