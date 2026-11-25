"""Drop rows that refer to images no longer in metadata.jsonl from the review queues built in v1
(Stage 2 dedup and Stage 3 label queues are NOT recomputed: author's decision, PROGRESS E-26).
Usage: python -I scripts/filter_queues.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, ROOT, setup_console, wlog  # noqa: E402

QUEUES = [("dedup_candidates.csv", ("image_id_a", "image_id_b")),
          ("label_disagreements.csv", ("image_id",)),
          ("commons_region_hints.csv", ("image_id",))]


def main():
    setup_console()
    ids = {json.loads(line)["image_id"] for line in open(ROOT / "metadata.jsonl", encoding="utf-8")}
    out = []
    for name, cols in QUEUES:
        p = REPORTS / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        rows = list(csv.DictReader(open(p, encoding=enc)))
        if not rows:
            continue
        keep = [r for r in rows if all(r.get(c) in ids for c in cols if c in r)]
        with open(p, "w", encoding=enc, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)
        out.append(f"{name} {len(rows)} -> {len(keep)}")
        print(out[-1])
    wlog("Stage2/3", "复核队列剔除已删除的图（不重算去重与标签分歧，PROGRESS E-26）", "python -I scripts/filter_queues.py",
         len(ids), "；".join(out))


if __name__ == "__main__":
    main()
