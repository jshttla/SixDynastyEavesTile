"""Stage 6 quality flags -> reports/quality.csv (image_id, quality_flags, metrics).

Flags (measured on the std crop + mask when present, else on the working copy):
  blur            variance of Laplacian of the tile region (1024 px crop) < 25
  overexposed     >5 % of tile pixels at >= 250 in any channel
  underexposed    mean luminance of tile region < 40
  low_resolution  short side of the ORIGINAL < 600 px, or tile spans < 400 original px
  grayscale       mean |R-G|+|G-B| < 3 (monochrome photo / scan)
Usage: python -I scripts/quality.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import REPORTS, ROOT, read_jsonl, setup_console, wlog  # noqa: E402

Image.MAX_IMAGE_PIXELS = 400_000_000


def load(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def main():
    setup_console()
    recs = read_jsonl(ROOT / "metadata.jsonl")
    seg = {}
    sp = REPORTS / "seg_index.csv"
    if sp.exists():
        seg = {r["image_id"]: r for r in csv.DictReader(open(sp, encoding="utf-8"))}
    rows = []
    for r in recs:
        iid = r["image_id"]
        crop, mask = ROOT / "std" / f"{iid}_crop.jpg", ROOT / "std" / f"{iid}_mask.png"
        if crop.exists() and mask.exists():
            img, m = load(crop), cv2.imdecode(np.fromfile(str(mask), np.uint8), cv2.IMREAD_GRAYSCALE) == 255
        else:
            w = ROOT / "cloud" / "work" / "upload" / "images" / f"{iid}.jpg"
            img = load(w if w.exists() else ROOT / r["file_name"])
            s = 1024 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
            m = np.ones(img.shape[:2], bool)
        if m.sum() < 100:
            m = np.ones(img.shape[:2], bool)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_64F)
        er = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        lv = float(lap[er].var()) if er.sum() > 100 else float(lap.var())
        over = float((img[m].max(1) >= 250).mean())
        lum = float(g[m].mean())
        b, gg, rr = [img[..., k][m].astype(np.int16) for k in range(3)]
        chroma = float((np.abs(rr - gg) + np.abs(gg - b)).mean())
        short = min(r["width"], r["height"])
        span = None
        s = seg.get(iid)
        if s and s.get("ok") == "1" and s.get("crop_x1"):
            k = max(r["width"], r["height"]) / max(int(s.get("orig_w") or 1), int(s.get("orig_h") or 1))
            span = (int(s["crop_x1"]) - int(s["crop_x0"])) * k / 1.1
        flags = []
        if lv < 25:
            flags.append("blur")
        if over > 0.05:
            flags.append("overexposed")
        if lum < 40:
            flags.append("underexposed")
        if short < 600 or (span is not None and span < 400):
            flags.append("low_resolution")
        if chroma < 3:
            flags.append("grayscale")
        rows.append({"image_id": iid, "quality_flags": ";".join(flags), "lap_var": round(lv, 1), "over_frac": round(over, 4),
                     "mean_lum": round(lum, 1), "chroma": round(chroma, 2), "short_side": short,
                     "tile_span_px": round(span) if span else ""})
    with open(REPORTS / "quality.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(x for row in rows for x in row["quality_flags"].split(";") if x)
    wlog("Stage6", "质量标记（模糊 / 过曝 / 欠曝 / 分辨率不足 / 灰度），在 std 裁剪图的 tile_front 区域内计算",
         "python -I scripts/quality.py", len(rows), f"各标记计数 {dict(c)}；reports/quality.csv")
    print(dict(c))


if __name__ == "__main__":
    main()
