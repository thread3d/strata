#!/bin/sh
# metal/tests/q36_real_compare.sh - the same prompt, the same pack, this engine against llama.cpp.
#
# Why a script: the token comparison has to be exact to mean anything, and every hand-typed version of it differed in a way that
# mattered - a different prompt encoding, a different KV cap, a chat template on one side and not the other.  This pins the recipe:
#
#   1. llama.cpp TOKENIZES the prompt (this port has no merge table, and does not need one if the reference hands it the ids);
#   2. this engine runs those ids, greedily, with no template;
#   3. llama.cpp runs the same text, greedily, with no template - and its printed continuation is the reference.
#
# Usage:  q36_real_compare.sh PATH_TO_Q8_0_PACK [PROMPT] [N_NEW]
# Needs:  build-metal/strata-q36-metal and a llama.cpp build (LLAMA_BIN_DIR, default ~/projects/llama.cpp/build/bin).
set -e
MODEL=${1:?usage: q36_real_compare.sh PACK [PROMPT] [N]}
PROMPT=${2:-The capital of France is}
N=${3:-6}
HERE=$(cd "$(dirname "$0")" && pwd)
ENGINE=$HERE/../build-metal/strata-q36-metal
[ -x "$ENGINE" ] || ENGINE=$HERE/../../build-metal/strata-q36-metal
LLAMA=${LLAMA_BIN_DIR:-$HOME/projects/llama.cpp/build/bin}

echo "== the prompt's ids, from llama.cpp's own tokenizer =="
IDS=$("$LLAMA/llama-tokenize" -m "$MODEL" -p "$PROMPT" 2>/dev/null | awk '/-> /{printf "%s,", $1}' | sed 's/,$//')
echo "  $PROMPT  ->  $IDS"

echo "== this engine, on those ids =="
"$ENGINE" --model "$MODEL" --tokens "$IDS" --n "$N" --kv-cap 64 --text 2>&1 | tail -2

echo "== llama.cpp, on the same text (the reference) =="
"$LLAMA/llama-cli" -m "$MODEL" -ngl 0 -c 64 -n "$N" -p "$PROMPT" --temp 0 --no-warmup 2>&1 |
    tr -d '\r' | sed -n "/$PROMPT/,/Prompt:/p" | head -12

echo "== read the two: the first token disagreeing is the first layer that is wrong =="
