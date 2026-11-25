"""Stage 4 acceptance: executor-traced reference polygons on 20 random images (方案 Stage 4 验收, PROGRESS E-13).

Step 1  python -I scripts/mask_check.py --make    -> qa/maskcheck/<image_id>.jpg (16 zoomed edge patches per image)
        Each patch: 96x96 px of the 1024-px std crop around a mask-boundary point, upsampled x3; magenta = mask
        boundary; ticks along the outward normal every 4 px (labelled -24..+24; + = outside the mask).
Step 2  a human (tonight: the executor; tomorrow: the author re-checks 5) reads, for every patch, where the TRUE
        object edge crosses the tick line and writes the offset in px into reports/mask_check_offsets.csv
        (image_id, k, offset_px).
Step 3  python -I scripts/mask_check.py --score   -> reference polygon = boundary points moved by the offsets
        (periodic interpolation between the 16 angles); IoU(reference polygon, mask) per image ->
        reports/mask_iou_check.md
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import REPORTS, ROOT, read_jsonl, setup_console, utcnow, wlog  # noqa: E402

N_IMG, N_PTS, HALF, UP = 20, 16, 48, 3
OUTD = ROOT / "qa" / "maskcheck"


def boundary_points(mask: np.ndarray):
    cs, _ = cv2.findContours((mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    m = cv2.moments((mask == 255).astype(np.uint8))
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    pts = []
    for k in range(N_PTS):
        a = -math.pi + 2 * math.pi * k / N_PTS
        i = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - a))))))
        p = c[i]
        t = c[(i + 4) % len(c)] - c[(i - 4) % len(c)]
        n = np.array([t[1], -t[0]]) / (np.linalg.norm(t) + 1e-9)
        # make the normal point outward
        q = p + 3 * n
        if 0 <= int(q[1]) < mask.shape[0] and 0 <= int(q[0]) < mask.shape[1] and mask[int(q[1]), int(q[0])] == 255:
            n = -n
        pts.append((p, n, a))
    return pts, (cx, cy), c


def pick():
    recs = [r for r in read_jsonl(ROOT / "metadata.jsonl") if r.get("crop_file") and (ROOT / r["mask_file"]).exists()]
    random.Random(20261108).shuffle(recs)
    return recs[:N_IMG]


def make():
    OUTD.mkdir(parents=True, exist_ok=True)
    chosen = pick()
    for r in chosen:
        img = cv2.imdecode(np.fromfile(str(ROOT / r["crop_file"]), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(ROOT / r["mask_file"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        pts, _, contour = boundary_points(mask)
        pad = cv2.copyMakeBorder(img, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        tiles = []
        for k, (p, n, a) in enumerate(pts):
            x, y = int(round(p[0])) + HALF, int(round(p[1])) + HALF
            patch = pad[y - HALF:y + HALF, x - HALF:x + HALF].copy()
            patch = cv2.resize(patch, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            cc = (contour - [p[0] - HALF, p[1] - HALF]) * UP
            cv2.polylines(patch, [cc.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 1, cv2.LINE_AA)
            o = np.array([HALF * UP, HALF * UP], float)
            for d in range(-24, 25, 4):
                q = o + n * d * UP
                tvec = np.array([-n[1], n[0]])
                L = 7 if d % 8 == 0 else 4
                cv2.line(patch, tuple((q - tvec * L).astype(int)), tuple((q + tvec * L).astype(int)), (0, 255, 255), 1)
                if d % 8 == 0:
                    cv2.putText(patch, str(d), tuple((q + tvec * 10).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.line(patch, tuple((o - n * 24 * UP).astype(int)), tuple((o + n * 24 * UP).astype(int)), (0, 200, 255), 1)
            cv2.putText(patch, f"#{k}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            tiles.append(patch)
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, N_PTS, 4)]
        sheet = np.vstack(rows)
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUTD / f"{r['image_id']}.jpg"))
    p = REPORTS / "mask_check_offsets.csv"
    if not p.exists():
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "k", "offset_px", "reader", "note"])
    wlog("Stage4", "生成 mask 人工核查补丁（随机种子 20261108 抽 20 张，每张 16 个边界点，放大 3 倍）", "python -I scripts/mask_check.py --make",
         len(chosen), "qa/maskcheck/*.jpg；判读结果写 reports/mask_check_offsets.csv")
    print([r["image_id"] for r in chosen])


def score():
    offs = {}
    for row in csv.DictReader(open(REPORTS / "mask_check_offsets.csv", encoding="utf-8")):
        offs.setdefault(row["image_id"], {})[int(row["k"])] = float(row["offset_px"])
    lines = ["# Stage 4 验收：执行者人工判读的参考多边形 vs 自动 mask", "",
             f"生成时间 {utcnow()}。方法：随机抽 20 张 std 图（种子 20261108），在 mask 边界上等角度取 16 个点，裁出 96 px 边缘补丁并放大 3 倍；"
             "判读者沿外法线刻度读出真实器物边缘的位置（单位 px，正值 = 真实边缘在 mask 外侧）。参考多边形 = 边界点按判读偏移移动，"
             "相邻角度之间周期插值；再算参考多边形与 mask 的 IoU。判读分辨率约 1 px（1024 px 尺度），所以 IoU 的不确定度约 ±0.005。", "",
             "| image_id | 判读点数 | 平均 \\|offset\\| px | 最大 \\|offset\\| px | IoU |", "|---|---|---|---|---|"]
    ious = []
    for r in pick():
        o = offs.get(r["image_id"])
        if not o or len(o) < N_PTS:
            lines.append(f"| {r['image_id']} | {len(o or {})} | – | – | 未判读 |")
            continue
        mask = cv2.imdecode(np.fromfile(str(ROOT / r["mask_file"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        pts, (cx, cy), contour = boundary_points(mask)
        ang = np.arctan2(contour[:, 1] - cy, contour[:, 0] - cx)
        ks = np.array([-math.pi + 2 * math.pi * k / N_PTS for k in range(N_PTS)])
        ov = np.array([o[k] for k in range(N_PTS)])
        ext_a = np.concatenate([ks - 2 * math.pi, ks, ks + 2 * math.pi])
        ext_o = np.concatenate([ov, ov, ov])
        d = np.interp(ang, ext_a, ext_o)
        # move every contour point along its local outward normal by the interpolated offset
        t = np.roll(contour, -4, 0) - np.roll(contour, 4, 0)
        nrm = np.stack([t[:, 1], -t[:, 0]], 1) / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        probe = (contour + 3 * nrm).round().astype(int)
        inside = mask[np.clip(probe[:, 1], 0, mask.shape[0] - 1), np.clip(probe[:, 0], 0, mask.shape[1] - 1)] == 255
        nrm[inside] *= -1
        ref = contour + nrm * d[:, None]
        R = np.zeros_like(mask, np.uint8)
        cv2.fillPoly(R, [ref.round().astype(np.int32).reshape(-1, 1, 2)], 1)
        M = (mask == 255).astype(np.uint8)
        iou = float((R & M).sum() / max(1, (R | M).sum()))
        ious.append(iou)
        lines.append(f"| {r['image_id']} | {len(o)} | {np.mean(np.abs(ov)):.1f} | {np.max(np.abs(ov)):.0f} | {iou:.4f} |")
    if ious:
        lines += ["", f"**结果：n = {len(ious)}，IoU 中位数 {np.median(ious):.4f}，最小 {min(ious):.4f}，"
                      f"≥ 0.95 的 {sum(i >= 0.95 for i in ious)} / {len(ious)}。** 验收线为 IoU ≥ 0.95。",
                  "", "补丁图在 `qa/maskcheck/`（本地复核用，不进公开仓库）。作者明早抽 5 张复看：打开补丁图，对照 `reports/mask_check_offsets.csv` 里的判读。"]
    (REPORTS / "mask_iou_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    wlog("Stage4", "mask 人工核查打分（参考多边形 vs mask IoU）", "python -I scripts/mask_check.py --score", len(ious),
         f"IoU 中位数 {np.median(ious):.4f}，最小 {min(ious):.4f}，≥0.95：{sum(i >= 0.95 for i in ious)}/{len(ious)}" if ious else "无判读数据")


if __name__ == "__main__":
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.make:
        make()
    if a.score:
        score()
