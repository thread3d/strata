#!/usr/bin/env python3
"""Reads the engine's own DFlash2 dump (STRATA_DF2_DUMP), the one the Metal window writes.

    python3 tools/dflash2/read_engine_dump.py <dump.bin>

The layout is metal/src/engine/dflash2.inc's df2_dump_write (magic "DSF2"): an 8-int header, then the
context rows the drafter was given, the anchor and mask embeddings, the block's 8 rows, the selector's
hidden projection, the top-16 per draft row, the answer, the vocabulary map (only when the head is the
106299-row one) and the draft head's logits.  The vocabulary map is not flagged in the header, so it is
inferred from the file length.

This is the engine's format; tools/dflash2/check_metal_dump.py reads the torch port's "STAP" tap file
instead.  Nothing here re-implements the reference: it prints what the drafter saw and what it answered,
with the norms that show whether a buffer is real data or an unfilled one.
"""
import array
import math
import struct
import sys

DF2_ROWS, DF2_DRAFTS = 8, 7          # dflash2.inc:75


def rms(a):
    if not a:
        return 0.0
    return math.sqrt(sum(float(v) * v for v in a) / len(a))


def main(argv):
    path = argv[1] if len(argv) > 1 else "dump.bin"
    blob = open(path, "rb").read()
    magic, ver, s, anchor, nd, nt, n, mask_tok = struct.unpack_from("<8i", blob, 0)
    if magic != 0x32465344:
        sys.exit("%s: not an engine DFlash2 dump (magic %s)" % (path, hex(magic)))
    off = 32

    def take(count, code="f"):
        nonlocal off
        a = array.array(code)
        a.frombytes(blob[off:off + 4 * count])
        off += 4 * count
        return a

    taps = take(s * nt * n)
    anchor_emb = take(n)
    mask_emb = take(n)
    fn = take(DF2_ROWS * n)
    hsel = take(DF2_DRAFTS * 256)
    topval = take(DF2_DRAFTS * 16)
    toptok = take(DF2_DRAFTS * 16, "i")
    out_i = take(32, "i")
    use_map = len(blob) - off > DF2_DRAFTS * nd * 4
    vmap = take(nd, "i") if use_map else None
    logits = take(DF2_DRAFTS * nd)
    out_f = array.array("f")
    out_f.frombytes(out_i.tobytes())

    print("%s: DSF2 v%d, %d context rows, anchor %d, draft vocab %d, %d taps of %d, mask token %d"
          % (path, ver, s, anchor, nd, nt, n, mask_tok))
    print("vocabulary map: %s" % ("present (%d entries, first %s)" % (nd, list(vmap[:5])) if use_map else "absent"))

    row = nt * n
    print("\nthe context the drafter was given (per row, RMS over the %d taps):" % nt)
    for i in range(min(s, 6)):
        per_slot = [rms(taps[i * row + k * n: i * row + (k + 1) * n]) for k in range(nt)]
        print("  pos %d  tok %-7d slot RMS %s" % (i, i, " ".join("%.4f" % v for v in per_slot)))
    allv = taps
    nz = sum(1 for v in allv if v == 0.0)
    nan = sum(1 for v in allv if v != v)
    print("  whole tap block: %d values, %d zero, %d NaN, RMS %.4f" % (len(allv), nz, nan, rms(allv)))

    print("\nembeddings: anchor RMS %.4f, mask RMS %.4f" % (rms(anchor_emb), rms(mask_emb)))
    print("block rows (fn): %d x %d, RMS %.4f; selector hidden (hsel): %d x 256, RMS %.4f"
          % (DF2_ROWS, n, rms(fn), DF2_DRAFTS, rms(hsel)))

    print("\nthe draft head's top-16 per row (mapped to target ids when the map is present):")
    for i in range(DF2_DRAFTS):
        vals = [(topval[i * 16 + j], toptok[i * 16 + j]) for j in range(16)]
        vals.sort(key=lambda t: -t[0])
        print("  row %d  %s" % (i, " ".join("%d(%.3f)" % (t, p) for p, t in vals[:5])))

    print("\nthe answer of the round:")
    print("  proposals  %s" % [out_i[i] for i in range(DF2_DRAFTS)])
    print("  confidence %s" % " ".join("%.3f" % out_f[16 + i] for i in range(DF2_DRAFTS)))
    print("  head argmax per row  %s"
          % [max(range(nd), key=lambda j: logits[i * nd + j]) for i in range(DF2_DRAFTS)])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
