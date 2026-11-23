#!/usr/bin/env python3
"""Compare two engine dumps: how much the block rows (fn) and the head's logits moved.

    python3 cmp_dumps.py a.bin b.bin

If the taps changed (different layers) the block rows must move; if the logits move far less than the
rows do, the head's GEMM is insensitive to its input, which would explain context-independent proposals.
"""
import struct
import sys

import numpy as np

DF2_ROWS, DF2_DRAFTS = 8, 7


def parse(path):
    raw = open(path, "rb").read()
    magic, ver, s, anchor, nd, nt, n, mask = struct.unpack_from("<8i", raw, 0)
    assert magic == 0x32465344, (path, hex(magic))
    off = 32

    def take(cnt, dt=np.float32):
        nonlocal off
        a = np.frombuffer(raw, dt, cnt, off)
        off += a.nbytes
        return a

    taps = take(s * nt * n).reshape(s, nt * n)
    take(n)
    take(n)
    fn = take(DF2_ROWS * n).reshape(DF2_ROWS, n)
    take(DF2_DRAFTS * 256)
    take(DF2_DRAFTS * 16)
    take(DF2_DRAFTS * 16, np.int32)
    take(32, np.int32)
    has_map = (len(raw) - off) > DF2_DRAFTS * nd * 4
    take(nd, np.int32) if has_map else None
    logits = take(DF2_DRAFTS * nd).reshape(DF2_DRAFTS, nd)
    return taps, fn, logits


def cos(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main(pa, pb):
    ta, fa, la = parse(pa)
    tb, fb, lb = parse(pb)
    print("%s  vs  %s" % (pa, pb))
    for i in range(ta.shape[0]):
        print("  tap row %d: cos %.6f  rms %.4f / %.4f" % (i, cos(ta[i], tb[i]), np.sqrt((ta[i] ** 2).mean()), np.sqrt((tb[i] ** 2).mean())))
    for i in range(DF2_ROWS):
        print("  block row %d: cos %.6f  rms %.4f / %.4f" % (i, cos(fa[i], fb[i]), np.sqrt((fa[i] ** 2).mean()), np.sqrt((fb[i] ** 2).mean())))
    for i in range(DF2_DRAFTS):
        a, b = la[i], lb[i]
        # remove each row's own mean so the correlation is not dominated by a common offset
        ca, cb = a - a.mean(), b - b.mean()
        print("  logits row %d: cos %.6f (mean-centred %.6f)  sd %.4f / %.4f  argmax %d / %d  |d| rms %.4f"
              % (i, cos(a, b), cos(ca, cb), a.std(), b.std(), int(a.argmax()), int(b.argmax()), float(np.sqrt(((a - b) ** 2).mean()))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
