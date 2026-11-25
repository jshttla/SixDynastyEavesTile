"""Merge the eye-review outputs (reports/review_v3/out_*.tsv, plus recheck_*.tsv for re-segmented masks) into:
  reports/review_v3/decisions.csv    image_id, decision (keep/delete), category, reason, ... -> build_metadata exclusions
  reports/review_v3/mask_flags.csv   latest mask verdict per image (ok/minor/wrong/none) -> quality_flags
  reports/review_v3/label_issues.csv reviewer remarks on museum labels -> author's QA (labels are NOT changed)
Rules (PROGRESS E-28): delete if is_wadang = no (not_wadang) or value = no (no_research_value); a wrong mask is flagged,
never a reason to delete. A recheck row (same columns, only mask fields used) overrides the first-pass mask verdict.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RV = ROOT / "reports" / "review_v3"


def read_tsv(pattern):
    rows = {}
    for p in sorted(RV.glob(pattern)):
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            if r.get("image_id"):
                rows[r["image_id"].strip()] = r
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    first = read_tsv("out_*.tsv")
    recheck = read_tsv("recheck_*.tsv")       # round 1 (v3.1 masks)
    recheck.update(read_tsv("recheck2_*.tsv"))  # round 2 (v3.2 masks) overrides round 1
    meta = {json.loads(line)["image_id"]: json.loads(line) for line in open(ROOT / "metadata.jsonl", encoding="utf-8")}
    seg = {r["image_id"] for r in csv.DictReader(open(ROOT / "reports" / "seg_index.csv", encoding="utf-8"))}
    missing = sorted(seg - set(first))
    dec, flags, issues = [], [], []
    for iid, r in sorted(first.items()):
        m = meta.get(iid, {})
        isw, val = (r.get("is_wadang") or "").strip(), (r.get("value") or "").strip()
        if isw == "no":
            d, cat, why = "delete", "not_wadang", r.get("what", "")
        elif val == "no":
            d, cat, why = "delete", "no_research_value", r.get("value_note", "")
        else:
            d, cat, why = "keep", "", ""
        dec.append({"image_id": iid, "decision": d, "category": cat, "reason": why, "source": m.get("source"),
                    "title_raw": m.get("title_raw")})
        mk = recheck.get(iid, r)
        flags.append({"image_id": iid, "mask": (mk.get("mask") or "").strip(), "mask_note": mk.get("mask_note", ""),
                      "pass": "recheck" if iid in recheck else "first"})
        if (r.get("label_issue") or "").strip():
            issues.append({"image_id": iid, "label_issue": r["label_issue"].strip(), "museum_motif": m.get("motif"),
                           "museum_object_type": m.get("object_type"), "title_raw": m.get("title_raw"),
                           "source_url": m.get("source_url")})
    for name, rows in (("decisions.csv", dec), ("mask_flags.csv", flags), ("label_issues.csv", issues)):
        with open(RV / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
            w.writeheader()
            w.writerows(rows)
    keep = {x["image_id"] for x in dec if x["decision"] == "keep"}
    print("reviewed", len(first), "| not yet reviewed", len(missing), "| recheck rows", len(recheck))
    print("decision", dict(Counter((x["decision"], x["category"]) for x in dec)))
    print("mask among kept", dict(Counter(x["mask"] for x in flags if x["image_id"] in keep)))
    print("label issues", len(issues))


if __name__ == "__main__":
    main()
