"""bench/macpro51/verify_page.py - recompute the matrix page's numbers from results-qwen38.jsonl and compare.

Run it after a sweep: it re-derives n, mean and population sd of the steady-state decode rate per pack (with the
answer gate applied) and reports any table row that disagrees by more than 0.005.  Exit code 1 on a mismatch."""
import json, os, re, statistics, sys
RES = os.path.expanduser("~/strata-bench/bench/macpro51/results-qwen38.jsonl")
DOC = os.path.expanduser("~/strata-bench/docs/macpro51-qwen38-benchmarks.md")
EXPECT = [271, 248068, 198, 760]
rows = [json.loads(l) for l in open(RES) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["label"], []).append(r)
def rate(r, which):
    d = r.get("decode_s") if which == "decode" else r.get("prefill_s")
    n = r.get("prefill_tokens") if which == "prefill" else r.get("decode_tokens")
    return n / d if d else 0.0
ok = []
for lab, rs in by.items():
    good = [r for r in rs if r.get("rc") == 0 and r.get("prefill_s") and (r.get("generated") or [])[:4] == EXPECT]
    steady = [r for r in good if r.get("round", 0) > 0]
    use = steady if len(steady) >= 2 else good
    if use:
        ok.append((lab, len(use), statistics.mean(x and rate(x, "decode") for x in use),
                   statistics.pstdev([rate(x, "decode") for x in use])))
doc = open(DOC).read()
bad = 0
for lab, n, m, sd in sorted(ok):
    # find the row in the decode table
    m1 = re.search(r"\| `%s` \| (\d+) \| ([\d.]+) ± ([\d.]+)" % re.escape(lab), doc)
    if not m1:
        print("MISSING in page:", lab); bad += 1; continue
    dn, dm, dsd = int(m1.group(1)), float(m1.group(2)), float(m1.group(3))
    if dn != n or abs(dm - m) > 0.005 or abs(dsd - sd) > 0.005:
        print("MISMATCH", lab, "page", dn, dm, dsd, "results", n, round(m, 4), round(sd, 4)); bad += 1
    else:
        print("ok", lab, n, round(m, 3), round(sd, 3))
print("checked", len(ok), "packs, problems", bad)
sys.exit(1 if bad else 0)
