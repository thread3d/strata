#!/usr/bin/env python3
"""Rank the target's true continuation tokens in the engine's DFlash2 draft head.

    python3 tools/dflash2/rank_true_token.py <dump.bin> [tok ...]

Reads the dump written by STRATA_DF2_DUMP (metal/src/engine/dflash2.inc, df2_dump_write) and, for each of
the seven draft rows, reports where the token the target actually emitted next ranks among the head's
`nd` logits.  A faithful drafter fed the right context should put it near the top; the selector can only
propose ids inside its top-16, so a truth outside the top-16 can never be accepted.

The tokens default to the prompt's documented continuation (Qwen3.8 UD-IQ1_M, prompt 3 1 42498 456 11):
76 398 8709 96302 17776 78 76.
"""
import struct
import sys

import numpy as np

DF2_ROWS, DF2_DRAFTS = 8, 7


def main(path, truth):
    raw = open(path, "rb").read()
    magic, ver, s, anchor, nd, nt, n, mask = struct.unpack_from("<8i", raw, 0)
    if magic != 0x32465344:
        sys.exit("%s: not an engine DFlash2 dump" % path)
    off = 32

    def take(cnt, dt=np.float32):
        nonlocal off
        a = np.frombuffer(raw, dt, cnt, off)
        off += a.nbytes
        return a

    take(s * nt * n)          # the context rows the drafter was given
    take(n)
    take(n)                   # anchor and mask embeddings
    take(DF2_ROWS * n)        # the block's rows
    take(DF2_DRAFTS * 256)    # the selector's hidden projection
    topval = take(DF2_DRAFTS * 16).reshape(DF2_DRAFTS, 16)
    toptok = take(DF2_DRAFTS * 16, np.int32).reshape(DF2_DRAFTS, 16)
    out_i = take(32, np.int32)
    out_f = np.frombuffer(out_i.tobytes(), np.float32)
    has_map = (len(raw) - off) > DF2_DRAFTS * nd * 4
    vmap = take(nd, np.int32) if has_map else None
    logits = take(DF2_DRAFTS * nd).reshape(DF2_DRAFTS, nd)

    print("%s: %d context rows, anchor %d, head %d rows%s" % (path, s, anchor, nd, " (mapped)" if has_map else ""))
    print("target continuation: " + " ".join(str(t) for t in truth))
    pos = {int(t): i for i, t in enumerate(vmap)} if has_map else None
    print()
    print(" row | engine pick (p) | head top-1 | true token | rank of true / %d | in the top-16" % nd)
    for i in range(DF2_DRAFTS):
        order = np.lexsort((np.arange(nd), -logits[i]))
        head1 = int(vmap[order[0]]) if has_map else int(order[0])
        pick = int(out_i[i])
        prob = float(out_f[16 + i]) if len(out_f) > 16 + i else float("nan")
        t = int(truth[i]) if i < len(truth) else -1
        if pos is not None:
            j = pos.get(t)
        else:
            j = t if 0 <= t < nd else None
        if j is None:
            print(" %3d | %10d (%.3f) | %10d | %10d | (not in the draft vocabulary)" % (i, pick, prob, head1, t))
            continue
        rank = int((logits[i] > logits[i][j]).sum()) + 1
        sel = np.where(toptok[i] == t)[0]
        print(" %3d | %10d (%.3f) | %10d | %10d | %10d | %s"
              % (i, pick, prob, head1, t, rank, ("yes, candidate %d" % int(sel[0])) if len(sel) else "no"))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "dump.bin"
    truth = [int(x) for x in sys.argv[2:]] or [76, 398, 8709, 96302, 17776, 78, 76]
    main(p, truth)
