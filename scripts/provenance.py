"""Provenance of one image, written once and reused everywhere: the sidecar .txt next to every image in the ZIP, the
metadata embedded in the derived standard crops (XMP) and masks (PNG text).
PROGRESS E-38 / E-39 (author, 2026-11-10): only what the licence asks for. Image number, the attribution exactly as the
licence requires it (only when it does; no licence name or link otherwise), the checksum, and this dataset. No title, no
museum, no accession number, no source link outside the required attribution. No RIS.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import OFFICIAL_LICENSE, ROOT

CFG = json.loads((ROOT / "release_config.json").read_text(encoding="utf-8"))
DERIVED_NOTE = "本文件由原图自动生成（裁剪、缩放、自动分割），可能有误。"
RECT_NOTE = "本文件由原图自动生成（裁剪、缩放、自动分割，并按瓦当外缘的椭圆拍正）。拍正为自动估计，可能不准。"  # E-55


def lic(r: dict) -> str:
    return OFFICIAL_LICENSE.get(r["license"], r["license"])


def credit(r: dict, derived: bool = False) -> str:
    """the attribution the licence requires (E-40: the derived line for crops and masks); '' when none is required"""
    if not r.get("attribution_required"):
        return ""
    return (r.get("attribution_text_derived") if derived else r.get("attribution_text")) or ""


def lines(r: dict, derived: bool = False, rect: bool = False) -> list[tuple[str, str]]:
    """(label, value) pairs; empty values are left out. rect: the rectified crop / mask (a derived file too)."""
    derived = derived or rect
    out = [("图像编号", r["image_id"]), ("署名", credit(r, derived)),  # E-39: no licence line; nothing unless required
           ("原图 SHA-256", r["sha256"]), ("说明", RECT_NOTE if rect else (DERIVED_NOTE if derived else "")),
           ("数据集", f"{CFG['dataset_title_zh']} v{CFG['version']}，{CFG['hf_url']}")]
    return [(a, b) for a, b in out if b]


def sidecar(r: dict, derived: bool = False, rect: bool = False) -> str:
    return "\n".join(f"{a}：{b}" for a, b in lines(r, derived, rect)) + "\n"


def xmp(r: dict, rect: bool = False) -> bytes:
    """XMP packet (UTF-8) for a derived standard crop (rect: for the rectified crop)."""
    from xml.sax.saxutils import escape as x
    rights = (f'<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{x(credit(r, True))}</rdf:li></rdf:Alt></dc:rights>\n'
              if credit(r, True) else "")
    pkt = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/">
<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{x(r['image_id'])}</rdf:li></rdf:Alt></dc:title>
{rights}<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{x(sidecar(r, derived=True, rect=rect))}</rdf:li></rdf:Alt></dc:description>
</rdf:Description></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>"""
    return pkt.encode("utf-8")


XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def embed_jpeg(path: Path, r: dict, rect: bool = False) -> None:
    """Insert / replace the XMP APP1 segment without re-encoding the image (lossless, idempotent)."""
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8", path
    i, segs = 2, []
    while i < len(data) and data[i] == 0xFF and data[i + 1] not in (0xDA, 0xD9):
        n = int.from_bytes(data[i + 2:i + 4], "big")
        seg = data[i:i + 2 + n]
        if not (data[i + 1] == 0xE1 and seg[4:4 + len(XMP_HEADER)] == XMP_HEADER):
            segs.append(seg)
        i += 2 + n
    payload = XMP_HEADER + xmp(r, rect)
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    pos = 1 if segs and segs[0][1] == 0xE0 else 0  # after JFIF APP0 if present
    segs.insert(pos, app1)
    path.write_bytes(b"\xff\xd8" + b"".join(segs) + data[i:])


def embed_png(path: Path, r: dict, rect: bool = False) -> None:
    """Rewrite the mask PNG (lossless) with UTF-8 text chunks; earlier text chunks are dropped."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    im = Image.open(path)
    im.load()
    info = PngInfo()
    info.add_itxt("Title", r["image_id"])
    if credit(r, True):
        info.add_itxt("Copyright", credit(r, True))
    info.add_itxt("Description", sidecar(r, derived=True, rect=rect))
    im.save(path, pnginfo=info, optimize=True)
