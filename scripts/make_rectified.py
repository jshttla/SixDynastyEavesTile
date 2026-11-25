"""Rectified ("拍正") versions of the standard crops (PROGRESS E-55; 《Q1 工作方案》 Stage 5, item 6).
For every image whose rim the geometry stage fitted as a valid ellipse (rim.csv: shape == round, not a rubbing), the
standard crop and its mask are stretched along the ellipse's minor axis by a/b (affine, rectification_method =
affine_ellipse), so that the fitted rim becomes a circle, and are framed again exactly like a standard crop (bounding
square of the tile + 5 %, 1024 px, grey outside the source frame).

The ellipse is the one the estimator used for the tilt angle (the inner rim if it passed its self-check, else the outer
rim). Its axis ratio and direction do not change under the scale + shift between the estimator's frame and the standard
crop, so they are taken from rim.csv as recorded; nothing is re-estimated here. Photographs carry no focal length, so a
true perspective correction is not recoverable; the weak-perspective (affine) stretch is the only one available.

This is an automatic estimate and may be wrong: hand-made tiles are not perfect circles, thick rims bias the fit, and a
mask that includes the tube behind the tile misleads it. The files are published with that caveat (author, E-55).

Input : rim.csv, std/<id>_crop.jpg, std/<id>_mask.png        (no dependency on metadata.jsonl)
Output: rect/<id>_crop.jpg, rect/<id>_mask.png, reports/rect_index.csv
Usage : python -I scripts/make_rectified.py
"""
from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, ROOT, setup_console, wlog  # noqa: E402

OUT = ROOT / "rect"
SIDE = 1024
PAD = 0.05  # same framing rule as cloud/segment.py square_crop
SS = 2  # the photograph is warped at 2x and reduced by area averaging (the stretch + re-framing can shrink it)
FIELDS = ["image_id", "rim_edge_used", "tilt_deg_ellipse", "stretch", "angle_deg", "mask_axis_ratio_before",
          "mask_axis_ratio_after", "rect_file", "rect_mask_file"]


def ellipse_used(r: dict) -> tuple[float, float, str] | None:
    """(a / b, major-axis direction in radians, 'inner' | 'outer') of the ellipse the tilt estimate used."""
    if r.get("ok") != "1" or r.get("shape") != "round" or not r.get("tilt_deg_ellipse") or r.get("tilt_source") == "rubbing":
        return None
    if r.get("rim_edge_used") == "inner" and r.get("a_in_px") and r.get("b_in_px"):
        a, b, ang = float(r["a_in_px"]), float(r["b_in_px"]), float(r["angle_in_deg"])
        edge = "inner"
    else:
        a, b, ang = float(r["a_px"]), float(r["b_px"]), float(r["angle_deg"])
        edge = "outer"
    if not (b > 0 and 1.0 <= a / b <= 2.0001):  # the geometry stage accepts b/a >= 0.5 only
        return None
    return a / b, math.radians(ang), edge


def axis_ratio(tile: np.ndarray) -> float | None:
    """minor / major of an ellipse fitted to the outline of the tile mask (a quick check, not the estimator)."""
    cs, _ = cv2.findContours(tile.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 20:
        return None
    (_, _), (w, h), _ = cv2.fitEllipse(c)
    return round(min(w, h) / max(w, h), 4) if max(w, h) > 0 else None


def rectify(iid: str, k: float, th: float) -> tuple[bytes, bytes, float | None, float | None] | None:
    cp, mp = ROOT / "std" / f"{iid}_crop.jpg", ROOT / "std" / f"{iid}_mask.png"
    if not (cp.exists() and mp.exists()):
        return None
    im = Image.open(cp)
    icc = im.info.get("icc_profile")
    img = np.asarray(im.convert("RGB"))
    mask = np.asarray(Image.open(mp).convert("L"))
    tile = mask == 255
    ys, xs = np.nonzero(tile)
    if len(xs) < 100:
        return None
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    C = R @ np.diag([1.0, k]) @ R.T  # stretch by k along the minor axis (-s, c); the major axis is unchanged
    P = np.stack([xs, ys], 1).astype(np.float64) @ C.T
    x0, y0 = P.min(0)
    x1, y1 = P.max(0) + 1
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    org = np.array([(x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2])
    sc = SIDE / side
    M = np.hstack([sc * C, (-sc * org)[:, None]])  # standard-crop pixel -> rectified pixel
    big = cv2.warpAffine(img, SS * M, (SS * SIDE, SS * SIDE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(128, 128, 128))
    out = cv2.resize(big, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    omask = cv2.warpAffine(mask, M, (SIDE, SIDE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bj = io.BytesIO()
    Image.fromarray(out).save(bj, "JPEG", quality=92, **({"icc_profile": icc} if icc else {}))
    bm = io.BytesIO()
    Image.fromarray(omask).save(bm, "PNG", optimize=True)
    return bj.getvalue(), bm.getvalue(), axis_ratio(tile), axis_ratio(omask == 255)


def main():
    setup_console()
    OUT.mkdir(exist_ok=True)
    rim = list(csv.DictReader(open(ROOT / "rim.csv", encoding="utf-8")))
    rows, keep = [], set()
    for r in rim:
        e = ellipse_used(r)
        if e is None:
            continue
        k, th, edge = e
        res = rectify(r["image_id"], k, th)
        if res is None:
            continue
        jpg, png, before, after = res
        iid = r["image_id"]
        for name, data in ((f"{iid}_crop.jpg", jpg), (f"{iid}_mask.png", png)):
            (OUT / name).write_bytes(data)
            keep.add(name)
        rows.append({"image_id": iid, "rim_edge_used": edge, "tilt_deg_ellipse": r["tilt_deg_ellipse"], "stretch": round(k, 4),
                     "angle_deg": round(math.degrees(th) % 180, 2), "mask_axis_ratio_before": before, "mask_axis_ratio_after": after,
                     "rect_file": f"rect/{iid}_crop.jpg", "rect_mask_file": f"rect/{iid}_mask.png"})
    for p in OUT.iterdir():  # nothing stale
        if p.name not in keep:
            p.unlink()
    with open(REPORTS / "rect_index.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = [x for x in rows if x["mask_axis_ratio_before"] and x["mask_axis_ratio_after"]]
    closer = sum(1 for x in ok if x["mask_axis_ratio_after"] >= x["mask_axis_ratio_before"] - 1e-9)
    wlog("Stage5", "拍正图（PROGRESS E-55）：按 rim.csv 中估计器所用椭圆的轴比与方向，对标准裁剪图与掩膜沿短轴做仿射拉伸并重新取景（1024 px）",
         "python -I scripts/make_rectified.py", len(rim),
         f"生成 {len(rows)} 组（外缘记录 {len(rim)} 行中圆形且椭圆有效、非拓片）；掩膜外廓轴比更接近 1 的 {closer} / {len(ok)}")
    print("rectified", len(rows), "of", len(rim), "| outline closer to a circle:", closer, "/", len(ok))


if __name__ == "__main__":
    main()
