"""Overlay of each published std crop with its mask contour (green = tile, red = occluder), for the author's eye check.
Output: qa/hr_seg/<image_id>.jpg (460 px) and qa/hr_seg_big/<image_id>.jpg (1024 px). Not published.
Usage: python -I scripts/make_seg_overlays.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, setup_console  # noqa: E402

SMALL, BIG = ROOT / "qa" / "hr_seg", ROOT / "qa" / "hr_seg_big"


def one(args):
    iid, crop, mask = args
    im = cv2.imdecode(np.fromfile(str(ROOT / crop), np.uint8), cv2.IMREAD_COLOR)
    mk = cv2.imdecode(np.fromfile(str(ROOT / mask), np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None or mk is None:
        return iid
    for val, col in ((255, (0, 220, 0)), (128, (0, 0, 255))):
        cs, _ = cv2.findContours((mk == val).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, col, 6)
    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(BIG / f"{iid}.jpg"))
    sm = cv2.resize(im, (460, 460), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(str(SMALL / f"{iid}.jpg"))
    return None


def main():
    setup_console()
    SMALL.mkdir(parents=True, exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    jobs = [(r["image_id"], r["crop_file"], r["mask_file"]) for r in read_jsonl(ROOT / "metadata.jsonl") if r.get("crop_file")]
    with ProcessPoolExecutor(6) as ex:
        bad = [x for x in ex.map(one, jobs, chunksize=16) if x]
    print("overlays", len(jobs), "failed", bad[:5])


if __name__ == "__main__":
    main()
