"""QA materials for the morning review (方案 Stage 3/4 + §9 checklist).

qa/thumbs/<image_id>.jpg              300 px thumbnails (also used by reports)
qa/motif_<code>_pNN.jpg               one mosaic series per motif class (each cell <= 300 px, labelled with image_id)
qa/faces_pNN.jpg                      ALL has_face candidates: museum has_face=true OR either model says has_face
qa/masks_pNN.jpg                      segmentation + rim overlays (from the cloud run), all images
qa/README.md                          index with counts
Usage: python -I scripts/make_qa.py [--overlay-dir cloud/work/out/overlay]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from common import REPORTS, ROOT, read_jsonl, setup_console, wlog  # noqa: E402

Image.MAX_IMAGE_PIXELS = 400_000_000
QA = ROOT / "qa"
CELL, COLS, ROWS = 300, 8, 6


def font(size=14):
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


F = font(13)


def thumb(rec) -> Image.Image:
    p = QA / "thumbs" / f"{rec['image_id']}.jpg"
    if p.exists():
        return Image.open(p).convert("RGB")
    wc = ROOT / "cloud" / "work" / "upload" / "images" / f"{rec['image_id']}.jpg"  # sRGB working copy (faster)
    im = ImageOps.exif_transpose(Image.open(wc if wc.exists() else ROOT / rec["file_name"]))
    im.draft("RGB", (CELL * 2, CELL * 2))
    im = im.convert("RGB")
    im.thumbnail((CELL, CELL))
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, quality=85)
    return im


def mosaic(items, out_stem: str, caption, cell_img) -> list[str]:
    files = []
    per = COLS * ROWS
    for k in range(math.ceil(len(items) / per)):
        chunk = items[k * per:(k + 1) * per]
        rows = math.ceil(len(chunk) / COLS)
        sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + 34)), "white")
        d = ImageDraw.Draw(sheet)
        for i, it in enumerate(chunk):
            x, y = (i % COLS) * CELL, (i // COLS) * (CELL + 34)
            im = cell_img(it)
            sheet.paste(im, (x + (CELL - im.width) // 2, y + 34 + (CELL - im.height) // 2))
            t1, t2 = caption(it)
            d.text((x + 3, y + 1), t1, fill="black", font=F)
            d.text((x + 3, y + 17), t2[:40], fill=(90, 90, 90), font=F)
            d.rectangle([x, y, x + CELL - 1, y + CELL + 33], outline=(210, 210, 210))
        name = f"{out_stem}_p{k + 1:02d}.jpg"
        sheet.save(QA / name, quality=80)
        files.append(name)
    return files


def vlm_faces() -> set:
    ids = set()
    for p in (REPORTS / "vlm").glob("*.jsonl"):
        for x in read_jsonl(p):
            if (x.get("result") or {}).get("has_face") is True:
                ids.add(x["image_id"])
    return ids


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-dir", default=str(ROOT / "cloud" / "work" / "out" / "overlay"))
    a = ap.parse_args()
    recs = read_jsonl(ROOT / "metadata.jsonl")
    QA.mkdir(exist_ok=True)
    for old in QA.glob("*_p[0-9][0-9].jpg"):
        old.unlink()
    for r in recs:
        thumb(r)
    cap = lambda r: (f"{r['image_id']} {r['source']} {r['label_source'][:11]}", r.get("title_raw") or "")
    index = ["# qa/ 拼图索引", "", "每格 ≤300 px，左上角是 image_id、来源和 label_source，第二行是馆方标题。", "",
             "| 拼图 | 内容 | 张数 | 文件 |", "|---|---|---|---|"]
    by = defaultdict(list)
    for r in recs:
        by[r["motif"]].append(r)
    for m in sorted(by):
        files = mosaic(by[m], f"motif_{m}", cap, thumb)
        index.append(f"| 母题 {m} | motif = {m} | {len(by[m])} | {', '.join(files)} |")
    vf = vlm_faces()
    faces = [r for r in recs if r["has_face"] is True or r["image_id"] in vf]
    faces.sort(key=lambda r: (r["has_face"] is not True, r["image_id"]))
    capf = lambda r: (f"{r['image_id']} {'馆方' if r['has_face'] else '仅模型'} {r['motif']}", r.get("title_raw") or "")
    files = mosaic(faces, "faces", capf, thumb)
    index.append(f"| 人脸候选（全看） | 馆方 has_face=true，或任一模型判为有脸 | {len(faces)} | {', '.join(files)} |")
    od = Path(a.overlay_dir)
    if od.exists():
        ov = [r for r in recs if (od / f"{r['image_id']}.jpg").exists()]

        def cell_ov(r):
            im = Image.open(od / f"{r['image_id']}.jpg").convert("RGB")
            im.thumbnail((CELL, CELL))
            return im
        files = mosaic(ov, "masks", lambda r: (f"{r['image_id']} {r['object_type']}", r.get("title_raw") or ""), cell_ov)
        index.append(f"| 分割与外圆叠加 | 绿线=tile_front 轮廓，红=occluder，品红=外缘椭圆，黄=内缘椭圆 | {len(ov)} | {', '.join(files[:3])} …（共 {len(files)} 页） |")
    (QA / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    wlog("Stage3", "生成 QA 拼图（每个母题一组；人脸候选全量；分割叠加图）", "python -I scripts/make_qa.py", len(recs),
         f"母题类 {len(by)} 组；人脸候选 {len(faces)} 张（馆方 {sum(1 for r in faces if r['has_face'])}，仅模型 {sum(1 for r in faces if not r['has_face'])}）；索引 qa/README.md")
    print("faces", len(faces), "motifs", {k: len(v) for k, v in by.items()})


if __name__ == "__main__":
    main()
