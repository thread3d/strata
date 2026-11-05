#!/usr/bin/env python3
"""Turn summary.csv + tegrastats.log + manifest.tsv into the markdown tables of the report."""
import csv
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "results", "jetson-agx-xavier"))


def load_summary():
    d = {}
    with open(os.path.join(R, "summary.csv")) as fh:
        for row in csv.DictReader(fh):
            d[(row["tool"], row["key"], row["metric"])] = row
    return d


def fmt(row, unit="", digits=1):
    """median [min-max] (CV%)"""
    if row is None:
        return "—"
    med, lo, hi, cv = float(row["median"]), float(row["min"]), float(row["max"]), float(row["cv_pct"])
    return f"{med:.{digits}f}{unit} [{lo:.{digits}f}–{hi:.{digits}f}] ({cv:.1f}%)"


def row_of(S, tool, key, metric):
    return S.get((tool, key, metric))


# ---- thermals within each run window ---------------------------------------------------------
TS_RX = re.compile(r"^([0-9.]+) .*?GR3D_FREQ (\d+)%.*?(?:GPU@([0-9.]+)C)?.*?CPU@([0-9.]+)C.*?thermal@([0-9.]+)C")


def thermals():
    samples = []
    path = os.path.join(R, "tegrastats.log")
    if not os.path.exists(path):
        return {}
    for line in open(path, errors="replace"):
        m = re.match(r"^([0-9.]+) (.*)$", line)
        if not m:
            continue
        t, rest = float(m.group(1)), m.group(2)
        g = re.search(r"GR3D_FREQ (\d+)%", rest)
        gpu = re.search(r"GPU@([0-9.]+)C", rest)
        cpu = re.search(r"CPU@([0-9.]+)C", rest)
        soc = re.search(r"thermal@([0-9.]+)C", rest)
        ram = re.search(r"RAM (\d+)/(\d+)MB", rest)
        samples.append((t, int(g.group(1)) if g else None, float(gpu.group(1)) if gpu else None,
                        float(cpu.group(1)) if cpu else None, float(soc.group(1)) if soc else None))
    per_tool = {}
    with open(os.path.join(R, "manifest.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["kind"] != "measured":
                continue
            t0, t1 = float(row["start"]), float(row["end"])
            win = [s for s in samples if t0 <= s[0] <= t1]
            if not win:
                continue
            g = [s[1] for s in win if s[1] is not None]
            gp = [s[2] for s in win if s[2] is not None]
            cp = [s[3] for s in win if s[3] is not None]
            so = [s[4] for s in win if s[4] is not None]
            per_tool.setdefault(row["tool"], []).append(
                (st.median(g) if g else None, max(g) if g else None,
                 max(gp) if gp else None, max(cp) if cp else None, max(so) if so else None, len(win)))
    out = {}
    for tool, runs in per_tool.items():
        def m(i):
            v = [r[i] for r in runs if r[i] is not None]
            return st.median(v) if v else None
        out[tool] = dict(gpu_util_med=m(0), gpu_util_max=max(r[1] for r in runs if r[1] is not None),
                         gpu_c_max=m(2), cpu_c_max=m(3), soc_c_max=m(4), samples=sum(r[5] for r in runs))
    return out


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    S = load_summary()
    T = thermals()
    P = []
    w = P.append

    w("### GPU memory read bandwidth (`dflash2_bw_bench`, 256 MB buffer)\n")
    rows = []
    for shape in ("loop", "unroll"):
        for blocks in (112, 896, 3584, 14336):
            r = row_of(S, "dflash2_bw_bench", f"{shape} {blocks} blocks", "GB/s")
            if r:
                rows.append((shape, blocks, fmt(r, " GB/s")))
    w(table(["probe", "blocks", "median [min–max] (CV, n=5)"], rows)); w("")

    w("### Dense Q8_0 GEMV at the engine's shapes (`q8_gemv_bench`)\n")
    rows = []
    matrices = []
    for (tool, key, metric) in S:
        if tool == "q8_gemv_bench" and " | " in key:
            m = key.split(" | ")[0]
            if m not in matrices:
                matrices.append(m)
    for m in matrices:
        cells = [m]
        for cols in ("1col", "2col", "3col", "4col", "6col"):
            gb = row_of(S, "q8_gemv_bench", f"{m} | {cols}", "GB/s")
            cells.append(f"{float(gb['median']):.0f}" if gb else "—")
        us = row_of(S, "q8_gemv_bench", f"{m} | 1col", "us")
        cells.append(f"{float(us['median']):.0f}" if us else "—")
        rows.append(cells)
    w(table(["matrix", "1 col GB/s", "2 cols", "3 cols", "4 cols", "6 cols", "1 col µs"], rows))
    w("")
    w("(GB/s falls as columns are added because the same weights are read once for every column; the 1-column row is\n"
      "the decode case the engine runs per token.)\n")

    w("### Dense projections, two kernel variants (`b6_mmvq_bench`, A = the engine's, B = the alternative)\n")
    rows = []
    cases = sorted({k for (t, k, m) in S if t == "b6_mmvq_bench" and k.endswith("cols1")})
    for key in cases:
        a = row_of(S, "b6_mmvq_bench", key, "A_ms")
        b = row_of(S, "b6_mmvq_bench", key, "B_ms")
        if a and b:
            rows.append((key.replace(" cols1", ""), fmt(a, " ms", 4), fmt(b, " ms", 4),
                         f"{100 * (float(b['median']) / float(a['median']) - 1):+.1f}%"))
    w(table(["case (1 column)", "A ms", "B ms", "B vs A"], rows)); w("")

    w("### Launch cost (`dflash2_launch_bench`, 100 chained kernels)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "dflash2_launch_bench"}):
        metric = "us_per_kernel" if "us_per_kernel" in {m for (t, k, m) in S if k == key} else "events_ms"
        rows.append((key, fmt(row_of(S, "dflash2_launch_bench", key, metric), " µs" if metric == "us_per_kernel" else " ms")))
    w(table(["shape", "median [min–max] (CV)"], rows)); w("")

    w("### Router chain per layer, captured graph (`router_chain_bench`)\n")
    rows = []
    for tool in ("router_chain_bench_T1", "router_chain_bench_T2", "router_chain_bench_T4"):
        for key in sorted({k for (t, k, m) in S if t == tool}):
            rows.append((tool.split("_")[-1].upper(), key, fmt(row_of(S, tool, key, "us_per_layer"), " µs")))
    w(table(["T", "composition", "per layer (median [min–max] CV)"], rows)); w("")

    w("### VRAM expert decode (`expdec_bench`, groups of 20 experts, `T` tokens)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "expdec_bench" and (k.endswith(" def") or k.endswith(" dec"))
                       and k.startswith("T1 ") or (t == "expdec_bench" and k.endswith(" def") and k.startswith("T2 "))}):
        pass
    keys = [k for (t, k, m) in S if t == "expdec_bench" and m == "all_us"]
    want = [k for k in keys if k.endswith(" def") and (k.startswith("T1 real") or k.startswith("T2 real"))]
    for key in sorted(set(want)):
        allus = row_of(S, "expdec_bench", key, "all_us")
        gb = row_of(S, "expdec_bench", key, "GB/s")
        rows.append((key, fmt(allus, " µs"), fmt(gb, " GB/s")))
    w(table(["cell (T shape gu/down choice)", "all kernels", "weights read"], rows)); w("")

    w("### Weight-bandwidth-limited expert decode, two formats (`nvfp4_decode_bench`)\n")
    rows = []
    for tool in ("nvfp4_decode_bench", "nvfp4_decode_bench_iq"):
        for key in sorted({k for (t, k, m) in S if t == tool}):
            gb = row_of(S, tool, key, "GB/s")
            pct = row_of(S, tool, key, "pct_of_peak")
            best = row_of(S, tool, key, "best_ms")
            rows.append((f"{tool.split('_')[-1] if 'iq' in tool else 'nvfp4'} {key}", fmt(best, " ms"),
                         fmt(gb, " GB/s"), f"{float(pct['median']):.0f}% of peak"))
    w(table(["case", "best", "weights read", "of DRAM peak"], rows)); w("")

    w("### Prompt-path routed experts (`mmq_expert_bench`)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "mmq_expert_bench"}):
        rows.append((key, fmt(row_of(S, "mmq_expert_bench", key, "ms"), " ms"),
                     fmt(row_of(S, "mmq_expert_bench", key, "TOPS"), " TOPS", 2)))
    w(table(["rows/expert, product", "time", "throughput"], rows)); w("")

    w("### Trellis (Gyro) decode kernels (`tq_pascal_bench`)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "tq_pascal_bench" and k.startswith("mmvq")}):
        rows.append((key.replace("mmvq ", ""), fmt(row_of(S, "tq_pascal_bench", key, "nc1_us"), " µs"),
                     fmt(row_of(S, "tq_pascal_bench", key, "nc8_us"), " µs")))
    w(table(["type and shape", "1 column", "8 columns"], rows)); w("")

    w("### QSA decode kernels at a 131k context (`qsa_rows_bench`)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "qsa_rows_bench"}):
        rows.append((key, fmt(row_of(S, "qsa_rows_bench", key, "T1_us"), " µs"),
                     fmt(row_of(S, "qsa_rows_bench", key, "T2_us"), " µs"),
                     fmt(row_of(S, "qsa_rows_bench", key, "T4_us"), " µs"),
                     fmt(row_of(S, "qsa_rows_bench", key, "per_row_us"), " µs")))
    w(table(["kernel", "T=1", "T=2", "T=4", "each extra row"], rows)); w("")

    w("### Volta prompt attention against the FP32 fallback (`qsa_prompt_attn_sm70_bench`, 32k context)\n")
    rows = []
    keys = sorted({k for (t, k, m) in S if t == "qsa_prompt_attn_sm70_bench" and m == "speedup"})
    for key in keys:
        if any(q in key for q in ("q2048", "q256")):
            rows.append((key, fmt(row_of(S, "qsa_prompt_attn_sm70_bench", key, "fp32_ms"), " ms"),
                         fmt(row_of(S, "qsa_prompt_attn_sm70_bench", key, "volta_ms"), " ms"),
                         fmt(row_of(S, "qsa_prompt_attn_sm70_bench", key, "speedup"), "×", 2)))
    w(table(["KV format, queries", "FP32 fallback", "Volta path", "speedup"], rows)); w("")

    w("### GDN prompt recurrence (`gdn_chunk_bench`)\n")
    rows = []
    for key in sorted({k for (t, k, m) in S if t == "gdn_chunk_bench"}):
        rows.append((key, fmt(row_of(S, "gdn_chunk_bench", key, "tokens_ms"), " ms"),
                     fmt(row_of(S, "gdn_chunk_bench", key, "chunk_fma_ms"), " ms"),
                     fmt(row_of(S, "gdn_chunk_bench", key, "chunk_wmma_ms"), " ms")))
    w(table(["prompt", "token-by-token", "chunked (FMA)", "chunked (wmma)"], rows)); w("")

    w("### Drafter kernels (`dflash2_*`)\n")
    rows = []
    for tool, m1, m2 in (("dflash2_small_bench", "us", None), ("dflash2_attn_bench", "us_per_call", "KV_GBs"),
                         ("dflash2_gemm_bench", "us", "GB/s")):
        for key in sorted({k for (t, k, m) in S if t == tool}):
            a = row_of(S, tool, key, m1)
            b = row_of(S, tool, key, m2) if m2 else None
            rows.append((tool.replace("dflash2_", "").replace("_bench", ""), key,
                         fmt(a, " µs"), fmt(b, " GB/s" if m2 == "GB/s" else "") if m2 else "—"))
    w(table(["bench", "case", "time", "weights read / K/V"], rows)); w("")

    w("### Thermals and clocks during the measured runs (median of the five windows)\n")
    rows = []
    for tool, d in sorted(T.items()):
        rows.append((tool, f"{d['gpu_util_med']:.0f}%", f"{d['gpu_util_max']:.0f}%",
                     f"{d['gpu_c_max']:.0f} °C" if d["gpu_c_max"] else "—",
                     f"{d['cpu_c_max']:.0f} °C" if d["cpu_c_max"] else "—",
                     f"{d['soc_c_max']:.0f} °C" if d["soc_c_max"] else "—"))
    w(table(["bench", "GPU busy (med)", "GPU busy (max)", "GPU max", "CPU max", "SoC max"], rows)); w("")

    print("\n".join(P))


if __name__ == "__main__":
    main()
