"""Rebuild std/<id>_crop.jpg locally from the sRGB working copies and the crop boxes in seg_index.csv.

Why: the cloud->local link is ~100 KB/s, so only masks + indexes are downloaded (16 MB instead of 640 MB). The crop is a
deterministic function of the working copy and the square box (cloud/segment.py, 'std outputs'); this repeats that code.
Usage: python -I scripts/regen_std_crops.py SEG_INDEX STD_DIR
"""
import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

Image.MAX_IMAGE_PIXELS = 400_000_000
ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "cloud" / "work" / "upload" / "images"
_srgb = ImageCms.createProfile("sRGB")
_srgb_bytes = ImageCms.ImageCmsProfile(_srgb).tobytes()


def load_rgb(path):  # identical to cloud/segment.py
    im = ImageOps.exif_transpose(Image.open(path))
    icc = im.info.get("icc_profile")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), _srgb, outputMode="RGB")
        except Exception:
            pass
    return im.convert("RGB") if im.mode != "RGB" else im


def one(r, std_dir: Path):
    iid = r["image_id"]
    im = load_rgb(WC / f"{iid}.jpg")
    W0, H0 = im.size
    x0, y0, x1, y1 = (int(r[k]) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
    side = x1 - x0
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(im.crop((max(0, x0), max(0, y0), min(W0, x1), min(H0, y1))), (max(0, -x0), max(0, -y0)))
    canvas.resize((1024, 1024), Image.LANCZOS).save(std_dir / f"{iid}_crop.jpg", quality=92, icc_profile=_srgb_bytes)
    return iid


def main():
    seg, std_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [r for r in csv.DictReader(open(seg, encoding="utf-8")) if r["ok"] == "1"]
    with ThreadPoolExecutor(8) as ex:
        n = sum(1 for _ in ex.map(lambda r: one(r, std_dir), rows))
    print("regenerated", n)


if __name__ == "__main__":
    main()
