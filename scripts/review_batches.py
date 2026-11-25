"""Split the 2714 v3 images into 10 round-robin batches for the eye-review agents (PROGRESS E-28)."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "review_v3" / "batches"


def clean(t) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(t or ""))[:80]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    for line in open(ROOT / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        R[r["image_id"]] = r
    S = list(csv.DictReader(open(ROOT / "reports" / "seg_index.csv", encoding="utf-8")))
    for b in range(10):
        with open(OUT / f"batch_{b:02d}.tsv", "w", encoding="utf-8", newline="") as f:
            f.write("image_id\tcomposite\tsource\tmuseum_object_type\tmuseum_motif\ttitle_raw\tseg\n")
            for k, s in enumerate(S):
                if k % 10 != b:
                    continue
                r = R[s["image_id"]]
                title = clean(r["title_raw"])
                if r["source"] == "commons":
                    title = "[Commons 文件名，不可靠] " + title
                seg = "ok" if s["ok"] == "1" else "failed"
                if s.get("suspect"):
                    seg += "(" + s["suspect"] + ")"
                comp = str(ROOT / "qa" / "review_v3" / "img" / f"{s['image_id']}.jpg")
                f.write("\t".join([s["image_id"], comp, r["source"], r["object_type"] or "", r["motif"] or "", title, seg]) + "\n")
    print([sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in sorted(OUT.glob("batch_*.tsv"))])


if __name__ == "__main__":
    main()
