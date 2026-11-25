"""Add a review priority to reports/dedup_candidates.csv (run after dedup.py). Nothing is merged.

A  pHash <= 2 or cosine >= 0.95  -> very likely the same photograph / same object (check first)
B  pHash <= 6 or cosine >= 0.90  -> possibly the same object (another view, crop, re-photograph)
C  everything else in the queue    -> similar-looking objects (often same mould / same type); informational
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, setup_console, wlog  # noqa: E402

setup_console()
p = REPORTS / "dedup_candidates.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))


def num(x, default):
    try:
        return float(x)
    except Exception:
        return default


for r in rows:
    ph, cs = num(r.get("phash_dist"), 99), num(r.get("cos_sim"), 0)
    r["priority"] = "A" if (ph <= 2 or cs >= 0.95) else ("B" if (ph <= 6 or cs >= 0.90) else "C")
    r["same_source"] = "yes" if r["source_a"] == r["source_b"] else "no"
rows.sort(key=lambda r: (r["priority"], -num(r.get("cos_sim"), 0)))
fields = ["priority", "same_source"] + [k for k in rows[0] if k not in ("priority", "same_source")]
with open(p, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
c = Counter(r["priority"] for r in rows)
cs = Counter((r["priority"], r["same_source"]) for r in rows)
wlog("Stage2", "去重复核队列分级（A：pHash≤2 或 cos≥0.95；B：pHash≤6 或 cos≥0.90；C：其余）；不合并任何候选", "python -I scripts/dedup_priority.py",
     len(rows), f"A {c['A']} / B {c['B']} / C {c['C']}；跨来源对：A {cs[('A', 'no')]}、B {cs[('B', 'no')]}、C {cs[('C', 'no')]}")
print(dict(c), dict(cs))
