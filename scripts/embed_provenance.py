"""Write provenance into the derived files (author, PROGRESS E-37; content per E-38): XMP in every standard crop (JPEG, lossless insert)
and UTF-8 text chunks in every mask (PNG, lossless). Original images are never touched. Idempotent.
Usage: python -I scripts/embed_provenance.py   (run after build_metadata.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, setup_console, wlog  # noqa: E402
from provenance import embed_jpeg, embed_png  # noqa: E402


def main():
    setup_console()
    recs = [r for r in read_jsonl(ROOT / "metadata.jsonl") if r.get("crop_file")]
    n_rect = 0
    for r in recs:
        embed_jpeg(ROOT / r["crop_file"], r)
        embed_png(ROOT / r["mask_file"], r)
        if r.get("rect_file"):  # E-55: the rectified pair says that it was rectified, and that this may be wrong
            embed_jpeg(ROOT / r["rect_file"], r, rect=True)
            embed_png(ROOT / r["rect_mask_file"], r, rect=True)
            n_rect += 1
    wlog("Stage7", "把溯源信息写进标准裁剪图（XMP）与掩膜（PNG 文本块），原图不动（PROGRESS E-37；E-38 起只写图号、许可、署名、校验和）",
         "python -I scripts/embed_provenance.py", len(recs), f"无损写入，可重复运行；其中拍正图 {n_rect} 组")
    print("embedded", len(recs), "rectified", n_rect)


if __name__ == "__main__":
    main()
