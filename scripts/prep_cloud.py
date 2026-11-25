"""Prepare GPU working copies for the cloud batch (Stage 2/4/5).

Working copy = original converted to sRGB RGB (ICC/alpha/CMYK/EXIF orientation handled), long side <= 3072 px,
JPEG q92. Std outputs are 1024 px, so this cap has no visible effect; it cuts the upload ~3x.
Only metadata.jsonl images (public sources) go to the cloud. Private images NEVER go through this script.

Usage: python -I scripts/prep_cloud.py  -> cloud/work/upload/{images/*.jpg, manifest.csv}
"""
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageCms, ImageOps  # noqa: E402

from common import ROOT, read_jsonl, setup_console, wlog  # noqa: E402

Image.MAX_IMAGE_PIXELS = 400_000_000
CAP = 3072
OUT = ROOT / "cloud" / "work" / "upload"
CLOUD_DIR = "/workspace/wadang/work/upload/images"
_srgb = ImageCms.createProfile("sRGB")


def to_srgb(path: Path) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path))
    icc = im.info.get("icc_profile")
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), _srgb, outputMode="RGB")
        except Exception:
            pass
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def main():
    setup_console()
    assert "data_local" not in str(ROOT / "metadata.jsonl")
    recs = read_jsonl(ROOT / "metadata.jsonl")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in recs:
        dst = OUT / "images" / f"{r['image_id']}.jpg"
        src = ROOT / r["file_name"]
        if not dst.exists():
            im = to_srgb(src)
            W, H = im.size
            s = min(1.0, CAP / max(W, H))
            if s < 1:
                im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
            im.save(dst, quality=92)
        with Image.open(dst) as w:
            ww, wh = w.size
        rows.append({"image_id": r["image_id"], "path": f"{CLOUD_DIR}/{r['image_id']}.jpg", "modality": r["modality"],
                     "orig_w": r["width"], "orig_h": r["height"], "work_w": ww, "work_h": wh})
    with open(OUT / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    size = sum(p.stat().st_size for p in (OUT / "images").glob("*.jpg"))
    wlog("Stage4", "生成云端工作副本：原图转 sRGB RGB（处理 ICC、alpha、EXIF 方向），长边限 3072 px，JPEG q92；std 输出为 1024 px，不受影响",
         "python -I scripts/prep_cloud.py", len(rows), f"cloud/work/upload/images 共 {size / 1e6:.0f} MB；manifest.csv {len(rows)} 行")
    print(len(rows), f"{size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
