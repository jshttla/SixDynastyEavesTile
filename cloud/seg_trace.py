"""Trace the PRODUCTION segmentation (segment.py v3.2, unchanged) on a few public images and draw every step.

How: Models is wrapped so that every detect / segment / salient call is recorded with its inputs and outputs, then
segment.process() is called exactly as in the batch run. The panels are drawn from that record, so they show the path the
real code took. Extra panel "no GDINO": BiRefNet alone on the whole working image (what we would get without the detector).
--ablation: the same "no GDINO" comparison for every listed image, against the published v3.2 masks (work/out_v32/seg).

Output: OUT/<image_id>/<step>.jpg (<= 460 px) + OUT/<image_id>/trace.json ; OUT/ablation.csv
Usage : venv/bin/python seg_trace.py work/manifest_v3.csv OUT --ids img_a,img_b [--ablation ids.txt --seg work/out_v32]
Public images only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
from PIL import Image

import segment as S

SIDE = 460


class TraceModels(S.Models):
    def __init__(self):
        super().__init__()
        self.log = []

    def detect(self, img, prompt, thr=0.2):
        r = super().detect(img, prompt, thr)
        self.log.append({"kind": "detect", "img": img, "prompt": prompt, "thr": thr, "out": r})
        return r

    def segment(self, img, box, pos=None, neg=None, return_all=False):
        r = super().segment(img, box, pos, neg, return_all)
        self.log.append({"kind": "segment", "img": img, "box": list(map(float, box)), "pos": pos, "neg": neg,
                         "all": return_all, "out": r})
        return r

    def salient(self, img):
        r = super().salient(img)
        self.log.append({"kind": "salient", "img": img, "out": r})
        return r


def rgb(img) -> np.ndarray:
    return np.asarray(img.convert("RGB")).copy()


def overlay(a: np.ndarray, m: np.ndarray, color, alpha=0.45, edge=True) -> np.ndarray:
    a = a.copy()
    if m is not None and m.any():
        c = np.array(color, np.float32)
        a[m] = (a[m] * (1 - alpha) + c * alpha).astype(np.uint8)
        if edge:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(a, cs, -1, tuple(int(v) for v in color), max(2, a.shape[1] // 300))
    return a


def box(a, b, color, t=None, label=None, dashed=False):
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    t = t or max(2, a.shape[1] // 260)
    if dashed:
        for (p, q) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = max(1, int(np.hypot(q[0] - p[0], q[1] - p[1]) // 18))
            for k in range(0, n, 2):
                s = (int(p[0] + (q[0] - p[0]) * k / n), int(p[1] + (q[1] - p[1]) * k / n))
                e = (int(p[0] + (q[0] - p[0]) * (k + 1) / n), int(p[1] + (q[1] - p[1]) * (k + 1) / n))
                cv2.line(a, s, e, color, t)
    else:
        cv2.rectangle(a, (x0, y0), (x1, y1), color, t)
    if label:
        fs = max(0.5, a.shape[1] / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        yy = max(th + 4, y0 - 4)
        cv2.rectangle(a, (x0, yy - th - 4), (x0 + tw + 4, yy + 2), color, -1)
        cv2.putText(a, label, (x0 + 2, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


def dot(a, p, color, r=None):
    r = r or max(5, a.shape[1] // 90)
    cv2.circle(a, (int(p[0]), int(p[1])), r, (255, 255, 255), -1)
    cv2.circle(a, (int(p[0]), int(p[1])), int(r * 0.7), color, -1)


def save(a: np.ndarray, path: str, side=SIDE):
    im = Image.fromarray(a)
    im.thumbnail((side, side), Image.LANCZOS)
    im.save(path, quality=80)


def area(d, W, H):
    return (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) / (W * H)


def full_mask_from_npz(z, size_work, scale):
    """published tile mask (pass-2 crop frame) -> working-image frame"""
    cx0, cy0, cx1, cy1 = [int(v) for v in z["crop_box"]]
    W0, H0 = [int(v) for v in z["orig_size"]]
    tile = z["tile"].astype(np.uint8)
    t_orig = cv2.resize(tile, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H0, W0), np.uint8)
    full[cy0:cy1, cx0:cx1] = t_orig[: H0 - cy0, : W0 - cx0]
    return cv2.resize(full, size_work, interpolation=cv2.INTER_NEAREST) > 0


def trace(models: TraceModels, iid: str, path: str, modality: str, out: str, prod_seg: str):
    d = os.path.join(out, iid)
    os.makedirs(d, exist_ok=True)
    models.log = []
    t0 = time.time()
    row = S.process(models, iid, path, modality, os.path.join(out, "_prod_like"))
    L = models.log
    work = L[0]["img"]
    Ww, Hw = work.size
    A = rgb(work)
    info = {"image_id": iid, "modality": modality, "row": row, "sec": round(time.time() - t0, 2)}
    save(A, f"{d}/p01_input.jpg")
    # --- GDINO tile prompt (first detect call), generic fallback if any
    det_calls = [x for x in L if x["kind"] == "detect" and x["img"] is work]
    tile_call = det_calls[0]
    generic = [x for x in det_calls if x["prompt"].startswith("terracotta object")]
    prop_call = [x for x in det_calls if x["prompt"] == S.PROP_PROMPT][0]
    a = A.copy()
    all_dets = tile_call["out"] + (generic[0]["out"] if generic else [])
    for b, sc, lab in sorted(all_dets, key=lambda z: z[1]):
        small = area((b, sc, lab), Ww, Hw) <= 0.02
        box(a, b, (150, 150, 150) if small else (0, 170, 60), label=f"{sc:.2f}", dashed=small)
    save(a, f"{d}/p02_gdino.jpg")
    info["gdino_prompt"] = tile_call["prompt"]
    info["gdino"] = [{"box": [round(v, 1) for v in b], "score": round(sc, 3), "label": lab,
                      "too_small": area((b, sc, lab), Ww, Hw) <= 0.02} for b, sc, lab in tile_call["out"]]
    info["generic_fallback"] = bool(generic)
    # --- props
    props = [p for p in prop_call["out"] if area(p, Ww, Hw) > 0.005]
    dets = [x for x in all_dets if area(x, Ww, Hw) > 0.02]
    skipped = [x for x in dets if any(S.box_iou(x[0], p[0]) > 0.6 and p[1] > x[1] for p in props)]
    a = A.copy()
    for b, sc, lab in props:
        box(a, b, (255, 140, 0), label=f"{lab.split()[0]} {sc:.2f}")
    for b, sc, lab in skipped:
        box(a, b, (220, 0, 0), label="skip", dashed=True)
    save(a, f"{d}/p03_props.jpg")
    info["props"] = [{"label": lab, "score": round(sc, 3)} for b, sc, lab in props]
    info["skipped_by_prop"] = len(skipped)
    # --- picked box = box given to the first SAM call on the working image
    seg_work = [x for x in L if x["kind"] == "segment" and x["img"] is work]
    keep = [x for x in dets if x not in skipped]
    pool = [x for x in keep if area(x, Ww, Hw) <= 0.9] or keep
    a = A.copy()
    best = max(pool, key=lambda z: z[1]) if pool else None
    for b, sc, lab in pool:
        box(a, b, (170, 170, 170))
    if best:
        box(a, best[0], (240, 200, 0), label=f"top {best[1]:.2f}")
    if seg_work:
        pb = seg_work[0]["box"]
        box(a, pb, (0, 200, 70), t=max(4, Ww // 160), label="picked")
        info["picked_box"] = [round(v, 1) for v in pb]
        info["shape_prior"] = row.get("shape_prior")
    save(a, f"{d}/p04_pick.jpg")
    # --- SAM pass 1 (coarse) ; BiRefNet rescue on the working image if it happened
    sal_work = [x for x in L if x["kind"] == "salient" and x["img"] is work]
    m1 = None
    if seg_work:
        m1 = S.largest_cc(seg_work[0]["out"][0])[0]
        info["sam_p1_score"] = round(seg_work[0]["out"][1], 3)
    a = overlay(A, m1, (0, 200, 70)) if m1 is not None else A.copy()
    if sal_work:
        a = overlay(a, S.largest_cc(sal_work[0]["out"])[0], (40, 120, 255), alpha=0.3)
    save(a, f"{d}/p05_sam_pass1.jpg")
    info["p1_rescue"] = row.get("rescue", "")
    # --- pass-2 crop with prompts
    seg_all = [x for x in L if x["kind"] == "segment" and x["all"]][0]
    crop = seg_all["img"]
    C = rgb(crop)
    cw, ch = crop.size
    a = C.copy()
    box(a, seg_all["box"], (40, 120, 255), label="box from pass 1")
    for p in seg_all["pos"] or []:
        dot(a, p, (0, 200, 70))
    for p in seg_all["neg"] or []:
        dot(a, p, (230, 30, 30))
    save(a, f"{d}/p06_crop_prompts.jpg")
    # --- BiRefNet on the crop, SAM's three granularities
    sal_crop = [x for x in L if x["kind"] == "salient" and x["img"] is crop][0]
    bn2 = S.fill_holes(S.largest_cc(sal_crop["out"])[0])
    save(overlay(C, bn2, (40, 120, 255)), f"{d}/p07_birefnet.jpg")
    cands, scores = seg_all["out"]
    ious_bn = [S.mask_iou(S.fill_holes(S.largest_cc(c)[0]), bn2) for c in cands]
    for k, c in enumerate(cands):
        save(overlay(C, c, (0, 200, 70)), f"{d}/p08_sam_cand{k}.jpg", side=300)
    info["sam_p2"] = [{"sam_score": round(s, 3), "iou_birefnet": round(i, 3)} for s, i in zip(scores, ious_bn)]
    info["sam_pick"] = row.get("sam_pick")
    # --- occluders + final
    occ_call = [x for x in L if x["kind"] == "detect" and x["img"] is crop and x["prompt"] == S.OCC_PROMPT]
    z = np.load(os.path.join(out, "_prod_like", "seg", f"{iid}.npz"))
    tile, occ = z["tile"], z["occ"]
    a = overlay(C, tile, (0, 200, 70))
    a = overlay(a, occ, (230, 0, 200), alpha=0.6)
    for b, sc, lab in (occ_call[0]["out"] if occ_call else []):
        box(a, b, (230, 0, 200), label=f"{lab.split()[-1]} {sc:.2f}", dashed=True)
    save(a, f"{d}/p09_final.jpg")
    info["occluder_detections"] = len(occ_call[0]["out"]) if occ_call else 0
    info["occluder_px"] = int(occ.sum())
    # --- sub-pixel edge: zoom on the top of the contour
    raw, sub = z["raw_contour"], z["sub_contour"]
    if len(raw):
        j = int(np.argmin(raw[:, 1]))
        cx, cy = raw[j]
        r = max(24, int(min(cw, ch) * 0.06))
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
        x1, y1 = int(min(cw, cx + r)), int(min(ch, cy + r))
        f = 8
        zoom = cv2.resize(C[y0:y1, x0:x1], ((x1 - x0) * f, (y1 - y0) * f), interpolation=cv2.INTER_NEAREST)
        for pts, col in ((raw, (230, 30, 30)), (sub, (0, 220, 80))):
            sel = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            q = ((pts[sel] - [x0, y0]) * f).astype(np.int32)
            for p in q:
                cv2.circle(zoom, (int(p[0]), int(p[1])), 3, col, -1)
        save(zoom, f"{d}/p10_subpixel.jpg", side=300)
        info["subpixel_shift_px_median"] = round(float(np.median(np.linalg.norm(sub - raw, axis=1))), 3)
    # --- std outputs + agreement with the published mask
    sc_ = Image.open(os.path.join(out, "_prod_like", "std", f"{iid}_crop.jpg"))
    sm = np.asarray(Image.open(os.path.join(out, "_prod_like", "std", f"{iid}_mask.png")))
    save(rgb(sc_), f"{d}/p11_std_crop.jpg", side=300)
    save(np.stack([sm] * 3, -1).astype(np.uint8), f"{d}/p12_std_mask.jpg", side=300)
    pm = os.path.join(prod_seg, "std", f"{iid}_mask.png")
    if os.path.exists(pm):
        pub = np.asarray(Image.open(pm)) == 255
        info["iou_with_published_mask"] = round(S.mask_iou(sm == 255, pub), 4)
    # --- what BiRefNet alone (no GDINO) gives on the whole working image
    bn_full = S.largest_cc(models.salient(work))[0]
    final_work = full_mask_from_npz(z, (Ww, Hw), None)
    save(overlay(A, bn_full, (40, 120, 255)), f"{d}/p13_no_gdino.jpg")
    info["no_gdino_iou_with_final"] = round(S.mask_iou(bn_full, final_work), 3)
    json.dump(info["row"], open(f"{d}/row.json", "w"))
    info.pop("row")
    info["row"] = {k: row.get(k) for k in ("det_label", "det_score", "n_det", "rescue", "bn_iou", "sam_pick", "shape_prior",
                                           "n_prop", "skipped_prop_boxes", "occluder_frac", "circularity", "work_scale")}
    json.dump(info, open(f"{d}/trace.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TRACED", iid, json.dumps(info["row"]), "iou_pub", info.get("iou_with_published_mask"), flush=True)


def ablation(models: S.Models, manifest: dict, ids: list[str], seg_dir: str, out_csv: str):
    """No-GDINO comparison on every image: BiRefNet on the whole working image vs the published v3.2 tile mask."""
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "iou", "bn_frac", "final_frac"])
        for i, iid in enumerate(ids):
            zp = os.path.join(seg_dir, "seg", f"{iid}.npz")
            if iid not in manifest or not os.path.exists(zp):
                continue
            im = S.load_rgb(manifest[iid]["path"])
            W0, H0 = im.size
            s = min(1.0, 2048 / max(W0, H0))
            work = im.resize((round(W0 * s), round(H0 * s)), Image.LANCZOS) if s < 1 else im
            bn = S.largest_cc(models.salient(work))[0]
            fin = full_mask_from_npz(np.load(zp), work.size, s)
            w.writerow([iid, round(S.mask_iou(bn, fin), 4), round(float(bn.mean()), 4), round(float(fin.mean()), 4)])
            if i % 100 == 0:
                print("ablation", i, len(ids), flush=True)
                f.flush()
    print("ABLATION_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--seg", default="work/out_v32", help="published v3.2 output dir (for the agreement check)")
    ap.add_argument("--ablation", default="", help="file with one image_id per line")
    a = ap.parse_args()
    man = {r["image_id"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    models = TraceModels()
    os.makedirs(a.out, exist_ok=True)
    for iid in [x for x in a.ids.split(",") if x]:
        r = man[iid]
        trace(models, iid, r["path"], r.get("modality", "photo"), a.out, a.seg)
    if a.ablation:
        ids = [x.strip() for x in open(a.ablation, encoding="utf-8") if x.strip()]
        ablation(models, man, ids, a.seg, os.path.join(a.out, "ablation.csv"))
    print("TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
