#!/bin/bash
# Prepare (repack + pack metadata + smoke) every Unsloth pack as soon as its download is complete.
# Runs in the background beside the downloader; skips packs already prepared.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MAN="$HERE/qwen38_unsloth_manifest.tsv"
LOG="${LOG:-$HOME/logs/prep-all-unsloth.log}"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }
PACKS=$(awk -F'\t' 'NR>1 && $2 ~ /^(unsloth|agentionai)\// {print $1}' "$MAN" | awk '!seen[$0]++')
log "packs: $(echo $PACKS)"
for pack in $PACKS; do
  D="$HOME/models/qwen38/$pack"
  if [ -f "$D/pack/native_experts.txt" ] && [ -f "$D/$pack-metal.gguf" ]; then log "SKIP  $pack (prepared)"; continue; fi
  # wait until every file of this pack has its exact size
  while :; do
    ready=1
    while IFS=$'\t' read -r p repo rev path bytes sha; do
      [ "$p" = "$pack" ] || continue
      f="$D/$(basename "$path")"
      have=$(stat -f%z "$f" 2>/dev/null || echo 0)
      [ "$have" = "$bytes" ] || { ready=0; break; }
    done < "$MAN"
    [ "$ready" = 1 ] && break
    sleep 120
  done
  log "PREP  $pack"
  bash "$HERE/prep_unsloth.sh" "$pack" > "$HOME/logs/prep-$pack.log" 2>&1
  log "DONE  $pack rc=$?"
done
log "ALL PREPARED"
