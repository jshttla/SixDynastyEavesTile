"""Select kept images whose mask must be re-checked after the v3.1 re-segmentation, and write recheck batches.

Masks of the two versions live in different std frames (different crop boxes), so both are mapped back to the working
image frame (crop box from each version's seg_index) before computing IoU.
Re-check = kept (decisions.csv) AND (IoU(v3, v3.1) < 0.9 OR first-pass mask in {wrong, minor, none} OR v3.1 failed).
Usage: python -I scripts/review_recheck.py OLD_STD OLD_SEG NEW_STD NEW_SEG N_BATCHES [--latest]
  --latest: re-check only masks that changed, or whose latest verdict is wrong/none (minor + unchanged carries over)
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RV = ROOT / "reports" / "review_v3"


def to_work(std_dir: Path, row: dict, W: int, H: int):
    mp = std_dir / f"{row['image_id']}_mask.png"
    if row.get("ok") != "1" or not mp.exists():
        return None
    m = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_GRAYSCALE) == 255
    x0, y0, x1, y1 = (int(row[k]) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
    side = x1 - x0
    big = cv2.resize(m.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0
    out = np.zeros((H, W), bool)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    w, h = min(W - dx0, side - sx0), min(H - dy0, side - sy0)
    if w > 0 and h > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = big[sy0:sy0 + h, sx0:sx0 + w]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    old_std, old_seg, new_std, new_seg, nb = Path(args[0]), args[1], Path(args[2]), args[3], int(args[4])
    O = {r["image_id"]: r for r in csv.DictReader(open(old_seg, encoding="utf-8"))}
    N = {r["image_id"]: r for r in csv.DictReader(open(new_seg, encoding="utf-8"))}
    man = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "cloud" / "work" / "manifest_v3.csv", encoding="utf-8"))}
    keep = {r["image_id"] for r in csv.DictReader(open(RV / "decisions.csv", encoding="utf-8-sig")) if r["decision"] == "keep"}
    # latest verdict per image (first pass, or the most recent recheck) -- see review_merge.py
    first = {r["image_id"]: r["mask"] for r in csv.DictReader(open(RV / "mask_flags.csv", encoding="utf-8-sig"))}
    bad = ("wrong", "none") if "--latest" in sys.argv else ("wrong", "minor", "none")
    rows, stats = [], {"kept": len(keep), "changed": 0, "first_bad": 0, "new_failed": 0}
    for iid in sorted(keep):
        W, H = int(man[iid]["work_w"]), int(man[iid]["work_h"])
        a, b = to_work(old_std, O.get(iid, {}), W, H), to_work(new_std, N.get(iid, {}), W, H)
        iou = float((a & b).sum()) / max(1, (a | b).sum()) if a is not None and b is not None else None
        why = []
        if b is None:
            why.append("v31_failed"); stats["new_failed"] += 1
        elif iou is None or iou < 0.9:
            why.append("changed"); stats["changed"] += 1
        if first.get(iid) in bad:
            why.append("first_" + first[iid]); stats["first_bad"] += 1
        if why:
            rows.append({"image_id": iid, "iou_v3_v31": "" if iou is None else round(iou, 3), "why": "+".join(why)})
    with open(RV / ("recheck2_selection.csv" if "--latest" in sys.argv else "recheck_selection.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "iou_v3_v31", "why"])
        w.writeheader()
        w.writerows(rows)
    bd = RV / ("recheck2_batches" if "--latest" in sys.argv else "recheck_batches")
    bd.mkdir(exist_ok=True)
    for k in range(nb):
        with open(bd / f"{bd.name}_{k:02d}.tsv", "w", encoding="utf-8", newline="") as f:
            f.write("image_id\tcomposite\n")
            for j, r in enumerate(rows):
                if j % nb == k:
                    f.write(f"{r['image_id']}\t{ROOT / 'qa' / 'review_v3' / 'img' / (r['image_id'] + '.jpg')}\n")
    print(stats, "| recheck", len(rows))


if __name__ == "__main__":
    main()
