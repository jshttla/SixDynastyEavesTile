"""Per-image review composites for the eye-review agents (PROGRESS E-28). Public images only.

Left = sRGB working copy (whole picture); right = v3 std crop with the tile_front contour (green) and occluder contour
(red); failed segmentations show a grey panel. Caption bar = image_id. 704x376 JPEG.
Output: qa/review_v3/img/<image_id>.jpg (local, not committed: third-party images)
Usage: python -I scripts/review_composites.py
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, setup_console  # noqa: E402

Image.MAX_IMAGE_PIXELS = 400_000_000
P, CAP = 352, 24
OUT = ROOT / "qa" / "review_v3" / "img"
WC = ROOT / "cloud" / "work" / "upload" / "images"
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)


def fit(im: Image.Image) -> Image.Image:
    im = ImageOps.contain(im, (P, P), Image.LANCZOS)
    c = Image.new("RGB", (P, P), (255, 255, 255))
    c.paste(im, ((P - im.width) // 2, (P - im.height) // 2))
    return c


def one(r: dict) -> str:
    iid = r["image_id"]
    out = Image.new("RGB", (2 * P, P + CAP), (255, 255, 255))
    out.paste(fit(Image.open(WC / f"{iid}.jpg").convert("RGB")), (0, CAP))
    d = ImageDraw.Draw(out)
    if r["ok"] == "1":
        im = cv2.imdecode(np.fromfile(str(ROOT / "std" / f"{iid}_crop.jpg"), np.uint8), cv2.IMREAD_COLOR)
        mk = cv2.imdecode(np.fromfile(str(ROOT / "std" / f"{iid}_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
        for val, col in ((255, (0, 230, 0)), (128, (0, 0, 255))):
            cs, _ = cv2.findContours((mk == val).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, 7)
        out.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (P, P), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (P, CAP))
    else:
        d.rectangle([P, CAP, 2 * P, CAP + P], fill=(150, 150, 150))
        d.text((P + 20, CAP + P // 2 - 10), "分割失败（无 mask）", fill="white", font=FONT)
    d.text((4, 2), f"{iid}    左：原图    右：标准裁剪 + mask（绿=瓦当，红=遮挡物）", fill="black", font=FONT)
    out.save(OUT / f"{iid}.jpg", quality=85)
    return iid


def main():
    setup_console()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "reports" / "seg_index.csv", encoding="utf-8")))
    with ThreadPoolExecutor(8) as ex:
        n = sum(1 for _ in ex.map(one, rows))
    print("composites", n)


if __name__ == "__main__":
    main()
