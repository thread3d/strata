#!/usr/bin/env python3
"""Aggregate the benchmark runs into per-metric statistics.

Reads bench/results/jetson-agx-xavier/runs/<tool>/run??.txt (run00 is the discarded warm-up) and writes
  metrics.csv  one row per (tool, key, metric, run)
  summary.csv  one row per (tool, key, metric): n, median, min, max, mean, stdev, cv_pct, iqr

Each tool below has its own table of regexes, taken from the format string the tool prints, so the report's tables come
from understood fields rather than from guessing which number on a line means what.
"""
import csv
import glob
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "results", "jetson-agx-xavier"))
RUNS = os.path.join(ROOT, "runs")
F = r"([0-9]+(?:\.[0-9]+)?)"

# tool -> (section regex or None, [(row regex, key builder, [(metric, group), ...]), ...])
# The section regex's group 1 prefixes every key until the next section line (used where one run prints several blocks
# whose rows would otherwise collide, e.g. one per matrix or per KV format).

def _kv(m):
    return m


TOOLS = {
    "dflash2_bw_bench": (None, [
        (re.compile(rf"^(loop|unroll)\s+blocks\s+(\d+):\s+{F} GB/s"),
         lambda m: f"{m.group(1)} {m.group(2)} blocks", [("GB/s", 3)]),
    ]),
    "dflash2_launch_bench": (None, [
        # "graph of 100 chained kernels,    1 blocks: 1.09 us per kernel"   (and the same for "stream")
        (re.compile(rf"^(graph|stream) of (\d+) chained kernels,\s+(\d+) blocks:\s+{F} us per kernel"),
         lambda m: f"{m.group(1)} {m.group(2)} kernels {m.group(3)} blocks", [("us_per_kernel", 4)]),
        # "single launch from idle, 100 kernels,    1 blocks: events 0.115 ms, wall 0.172 ms"
        (re.compile(rf"^single launch from idle, (\d+) kernels,\s+(\d+) blocks: events {F} ms, wall {F} ms"),
         lambda m: f"idle launch {m.group(1)} kernels {m.group(2)} blocks", [("events_ms", 3), ("wall_ms", 4)]),
    ]),
    "dflash2_gemm_bench": (None, [
        (re.compile(rf"^(\S.*?\S)\s+variant (\d+):\s+{F} us\s+{F} GB/s"),
         lambda m: f"{m.group(1)} v{m.group(2)}", [("us", 3), ("GB/s", 4)]),
    ]),
    "dflash2_small_bench": (None, [
        (re.compile(rf"^(\S.*?\S)\s+{F} us$"), lambda m: m.group(1), [("us", 2)]),
    ]),
    "dflash2_attn_bench": (None, [
        (re.compile(rf"^variant (\d+):\s+{F} us per call, K/V {F} GB/s"),
         lambda m: f"variant {m.group(1)}", [("us_per_call", 2), ("KV_GBs", 3)]),
    ]),
    "router_chain_bench": (None, [
        (re.compile(rf"^T (\d+) (\S.*?)\s+:\s+{F} us per layer"), lambda m: f"T{m.group(1)} {m.group(2)}",
         [("us_per_layer", 3)]),
    ]),
    "tq_pascal_bench": (None, [
        # "BENCH mmvq TQ2_T  8192 x 2560 (0.006 GB):  nc1 351.9 us (16 GB/s)  nc2 410.8 us (14 GB/s)  ..."
        (re.compile(rf"^BENCH mmvq (\S+)\s+(\d+) x\s+(\d+) \(([0-9.]+) GB\):\s+nc1 {F} us.*?nc2 {F} us.*?nc3 {F} us"
                    rf".*?nc4 {F} us.*?nc8 {F} us"),
         lambda m: f"mmvq {m.group(1)} {m.group(2)}x{m.group(3)}",
         [("nc1_us", 5), ("nc2_us", 6), ("nc3_us", 7), ("nc4_us", 8), ("nc8_us", 9)]),
        # "BENCH rows TQK6    8192 x  2560 nc1 347.6 us"
        (re.compile(rf"^BENCH rows (\S+)\s+(\d+) x\s+(\d+) nc1 {F} us"),
         lambda m: f"rows {m.group(1)} {m.group(2)}x{m.group(3)}", [("nc1_us", 4)]),
        # "BENCH grouped TQK6  /TQ2_T  16 groups of m entries:  m=1 772.8 us (23 GB/s)  ... m=8 2065.4 us (9 GB/s)"
        (re.compile(rf"^BENCH grouped (\S+)\s+/(\S+)\s+16 groups.*?m=1 {F} us.*?m=8 {F} us"),
         lambda m: f"grouped gate/up={m.group(1)} down={m.group(2)}",
         [("m1_us", 3), ("m8_us", 4)]),
    ]),
    "qsa_rows_bench": (None, [
        (re.compile(rf"^\s+(\w+)\s+(\S+)\s+.*?T=1\s+{F}\s+T=2\s+{F}\s+T=3\s+{F}\s+T=4\s+{F}\s+\|\s+\+{F} us a row"),
         lambda m: f"{m.group(1)} {m.group(2)}",
         [("T1_us", 3), ("T2_us", 4), ("T3_us", 5), ("T4_us", 6), ("per_row_us", 7)]),
    ]),
    "q8_gemv_bench": (re.compile(r"^(\S.*?):\s+[0-9.]+ MB"), [
        # "  1 col :    314.5 us     88.5 GB/s"   (singular, and a space before the colon)
        (re.compile(rf"^\s+(\d) col[s]?\s*:\s*{F} us\s+{F} GB/s"), lambda m: f"{m.group(1)}col",
         [("us", 2), ("GB/s", 3)]),
    ]),
    "gdn_chunk_bench": (None, [
        (re.compile(rf"^\s+(\d+)\s+{F}ms\s+{F}ms\s+{F}ms\s+{F}ms\s+{F}ms\s+([0-9.eE+-]+)"),
         lambda m: f"T{m.group(1)}",
         [("tokens_ms", 2), ("warp_ms", 3), ("fast_ms", 4), ("chunk_fma_ms", 5), ("chunk_wmma_ms", 6)]),
    ]),
    "qsa_prompt_attn_sm70_bench": (re.compile(r"^#\s*(\S.*?KV)"), [
        (re.compile(rf"^q\s+(\d+): fp32\s+{F} ms\s+volta\s+{F} ms\s+{F}x"), lambda m: f"q{m.group(1)}",
         [("fp32_ms", 2), ("volta_ms", 3), ("speedup", 4)]),
    ]),
    "mmq_expert_bench": (None, [
        (re.compile(rf"^\s+R\s+(\d+)\s+(\S.*?)\s+{F} ms\s+{F} TOPS"), lambda m: f"R{m.group(1)} {m.group(2)}",
         [("ms", 3), ("TOPS", 4)]),
    ]),
    "nvfp4_decode_bench": (None, [
        (re.compile(rf"^types (\d+)/(\d+).*best {F} ms\s+mean {F} ms.*?{F} GB/s of weights.*?\((\d+)% of the"),
         lambda m: f"types {m.group(1)}/{m.group(2)}",
         [("best_ms", 3), ("mean_ms", 4), ("GB/s", 5), ("pct_of_peak", 6)]),
    ]),
    "b6_mmvq_bench": (None, [
        # case cols A_ms B_ms A+col B+col bitwise
        (re.compile(rf"^(\S.*?)\s+(\d+)\s+{F}\s+{F}\s+{F}\s+{F}\s+ok"),
         lambda m: f"{m.group(1).strip()} cols{m.group(2)}", [("A_ms", 3), ("B_ms", 4)]),
    ]),
    "expdec_bench": (None, [
        # dev T shape gu/d choice all_us gu_us down_us GB/s pct% identical|DIFFERENT
        (re.compile(rf"^(\d+) (\d+) (\S+) (\d+/\d+) (\S+) {F} {F} {F} {F} {F}% (\S+)"),
         lambda m: f"T{m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)}",
         [("all_us", 6), ("gu_us", 7), ("down_us", 8), ("GB/s", 9), ("pct_of_peak", 10)]),
    ]),
}


def base_tool(dirname):
    if dirname.startswith("router_chain_bench"):
        return "router_chain_bench"
    if dirname.startswith("nvfp4_decode_bench"):
        return "nvfp4_decode_bench"
    return dirname


def parse(tool, text):
    section_rx, rows = TOOLS.get(tool, (None, []))
    out, section = [], ""
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if section_rx is not None:
            m = section_rx.match(line)
            if m:
                section = m.group(1).strip()
                if tool == "qsa_prompt_attn_sm70_bench":
                    continue                    # a section header, not a measurement
        for rx, keyf, metrics in rows:
            m = rx.match(line)
            if not m:
                continue
            key = keyf(m)
            if section:
                key = f"{section} | {key}"
            for name, gi in metrics:
                out.append((key, name, float(m.group(gi))))
            break
    return out


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"no runs directory at {RUNS}")
    rows, tools = [], sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
    for tool in tools:
        for path in sorted(glob.glob(os.path.join(RUNS, tool, "run*.txt"))):
            run = int(re.search(r"run(\d+)", path).group(1))
            if run == 0:
                continue                        # warm-up
            with open(path, errors="replace") as fh:
                for key, metric, value in parse(base_tool(tool), fh.read()):
                    rows.append((tool, key, metric, run, value))

    with open(os.path.join(ROOT, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool", "key", "metric", "run", "value"])
        w.writerows(rows)

    groups = {}
    for tool, key, metric, _run, value in rows:
        groups.setdefault((tool, key, metric), []).append(value)

    with open(os.path.join(ROOT, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool", "key", "metric", "n", "median", "min", "max", "mean", "stdev", "cv_pct", "iqr"])
        for (tool, key, metric), vals in sorted(groups.items()):
            n = len(vals)
            med, mean = st.median(vals), st.mean(vals)
            sd = st.stdev(vals) if n > 1 else 0.0
            q = st.quantiles(vals, n=4) if n > 1 else [med, med, med]
            w.writerow([tool, key, metric, n, f"{med:.6g}", f"{min(vals):.6g}", f"{max(vals):.6g}",
                        f"{mean:.6g}", f"{sd:.6g}", f"{100 * sd / mean if mean else 0.0:.2f}",
                        f"{q[2] - q[0]:.6g}"])

    print(f"{len(rows)} measurements -> {len(groups)} (tool, key, metric) series over {len(tools)} tools")
    for tool in tools:
        n = len({(k, m) for (t, k, m) in groups if t == tool})
        print(f"  {tool:<34} {n:>4} series")


if __name__ == "__main__":
    main()
