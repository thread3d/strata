#!/bin/bash
# Resumable, verified downloader for the Unsloth + Gyro Qwen3.8 packs on macpro51.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MAN="$HERE/qwen38_unsloth_manifest.tsv"
DESTROOT="$HOME/models/qwen38"
LOG="${LOG:-$HOME/logs/qwen38-download-unsloth.log}"
mkdir -p "$DESTROOT" "$(dirname "$LOG")"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }
fetch_one() {
  local pack="$1" repo="$2" rev="$3" path="$4" bytes="$5" sha="$6"
  local dir="$DESTROOT/$pack"; local name; name="$(basename "$path")"; local out="$dir/$name"
  local url="https://huggingface.co/$repo/resolve/$rev/$path?download=true"
  mkdir -p "$dir"
  if [ -f "$out" ]; then
    local have; have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$bytes" ]; then
      local got; got=$(shasum -a 256 "$out" | awk '{print $1}')
      if [ "$got" = "$sha" ]; then log "OK      $pack/$name (already complete, sha256 verified)"; return 0; fi
      log "CORRUPT $pack/$name (size ok, sha256 mismatch) - moving aside"; mv "$out" "$out.bad.$(date +%s)"
    fi
  fi
  log "START   $pack/$name ($bytes bytes)"
  local attempt=1
  while [ $attempt -le 12 ]; do
    curl -L --fail -C - --connect-timeout 30 --speed-time 120 --speed-limit 1024 -sS -o "$out" "$url" >>"$LOG" 2>&1
    local rc=$?; local have; have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    log "  attempt $attempt rc=$rc have=$have/$bytes"
    if [ "$have" = "$bytes" ]; then break; fi
    attempt=$((attempt+1)); sleep 5
  done
  local have; have=$(stat -f%z "$out" 2>/dev/null || echo 0)
  if [ "$have" != "$bytes" ]; then log "FAIL    $pack/$name ($have/$bytes)"; return 1; fi
  local got; got=$(shasum -a 256 "$out" | awk '{print $1}')
  if [ "$got" != "$sha" ]; then log "FAIL    $pack/$name sha256 mismatch"; return 1; fi
  log "DONE    $pack/$name sha256 verified"; return 0
}
rc=0
while IFS=$'\t' read -r pack repo rev path bytes sha; do
  [ "$pack" = "pack" ] && continue
  fetch_one "$pack" "$repo" "$rev" "$path" "$bytes" "$sha" || rc=1
done < "$MAN"
log "ALL DONE rc=$rc"
exit $rc
