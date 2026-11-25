"""Stage 4 segmentation v3 (cloud GPU). Grounding DINO (concept prompt -> box) + SAM 2.1 (box/points -> mask), two passes.

v3 = the v1 models (author's decision 2026-11-08 after the Falcon pre-test) with the three heuristics that had been tuned on
a private test set REMOVED: large-box penalty, centre-point/corner-negative retry, and pass-2 fallback to the
coarse mask (PROGRESS E-24/E-26). A swallowed mask (backdrop instead of object) is instead rescued by a different model,
BiRefNet (MIT; salient-object segmentation, built for single-object photos); if that fails the same check, the image is
recorded as failed (ok=0). Developed on public museum images only.

Input : manifest CSV with columns image_id,path,modality   (paths on the cloud disk)
Output: <out>/std/<image_id>_crop.jpg   square bbox crop, 1024 px, sRGB JPEG
        <out>/std/<image_id>_mask.png   same frame as crop; 0=background, 255=tile_front, 128=occluder
        <out>/seg/<image_id>.npz        pass-2 crop (<=2048 px) geometry: tile/occluder masks, raw + sub-pixel contour,
                                        crop box in original pixels, scale
        <out>/seg_index.csv             one row per image (see FIELDS)
Usage : venv/bin/python segment.py manifest.csv OUT_DIR [--limit N] [--prompt-test]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

Image.MAX_IMAGE_PIXELS = 400_000_000
# the container has a 24-CPU quota but reports 192 cores: cap threads per process, or parallel shards thrash (WORKLOG)
_NT = int(os.environ.get("NTHREADS", "6"))
torch.set_num_threads(_NT)
cv2.setNumThreads(_NT)
DEV = "cuda"
GDINO = "IDEA-Research/grounding-dino-base"
SAM = "facebook/sam2.1-hiera-large"
GDINO_REV = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
SAM_REV = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
BIREFNET = "ZhengPeng7/BiRefNet"
BIREFNET_REV = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
# v3.1: "eaves tile" removed -- it selected the neighbouring 軒平瓦 in paired photos (review W0360); 軒平瓦 is not a
# 瓦当 under the agreed definition (PROGRESS E-28)
TILE_PROMPT = "roof tile end. antefix."
RUB_PROMPT = "round ink rubbing. circular rubbing. roof tile end."
OCC_PROMPT = "metal stand. metal clamp. metal bracket. support rod."
# museum props that v1 mistook for the tile on public images (PROGRESS E-27, WORKLOG W0348)
PROP_PROMPT = "color checker. ruler. scale bar. label card. display case. glass shelf. pedestal. stand."
FIELDS = ["image_id", "ok", "error", "det_label", "det_score", "n_det", "rescue", "suspect", "bn_iou", "sam_pick", "shape_prior", "n_prop", "skipped_prop_boxes", "sam_iou_p1", "sam_iou_p2",
          "mask_frac", "n_components", "occluder_px", "occluder_frac", "touches_border", "circularity",
          "crop_x0", "crop_y0", "crop_x1", "crop_y1", "orig_w", "orig_h", "work_scale", "crop_file", "mask_file", "sec"]

_srgb = ImageCms.createProfile("sRGB")


def load_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im, src, _srgb, outputMode="RGB")
        except Exception:
            pass
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


class Models:
    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.gp = AutoProcessor.from_pretrained(GDINO, revision=GDINO_REV)
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO, revision=GDINO_REV).to(DEV).eval()
        self.sp = Sam2Processor.from_pretrained(SAM, revision=SAM_REV)
        self.sm = Sam2Model.from_pretrained(SAM, revision=SAM_REV, torch_dtype=torch.bfloat16).to(DEV).eval()
        self._bn = None

    @torch.no_grad()
    def salient(self, img: Image.Image) -> np.ndarray:
        """BiRefNet salient-object mask at the image's own size (rescue only)."""
        if self._bn is None:
            from transformers import AutoModelForImageSegmentation
            self._bn = AutoModelForImageSegmentation.from_pretrained(BIREFNET, revision=BIREFNET_REV,
                                                                     trust_remote_code=True).to(DEV).eval().half()
        x = np.asarray(img.resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEV).half()
        pred = self._bn(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return cv2.resize(pred, img.size, interpolation=cv2.INTER_LINEAR) > 0.5

    @torch.no_grad()
    def detect(self, img: Image.Image, prompt: str, thr: float = 0.2):
        inp = self.gp(images=img, text=prompt, return_tensors="pt").to(DEV)
        out = self.gm(**inp)
        r = self.gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=thr,
                                                           target_sizes=[img.size[::-1]])[0]
        labels = r.get("text_labels", r.get("labels"))
        return [(b.tolist(), float(s), str(l)) for b, s, l in zip(r["boxes"].cpu(), r["scores"].cpu(), labels)]

    @torch.no_grad()
    def segment(self, img: Image.Image, box, pos=None, neg=None, return_all=False):
        kw = {"input_boxes": [[list(map(float, box))]]}
        pts, lbl = [], []
        for p in pos or []:
            pts.append(list(map(float, p))); lbl.append(1)
        for p in neg or []:
            pts.append(list(map(float, p))); lbl.append(0)
        if pts:
            kw["input_points"] = [[pts]]
            kw["input_labels"] = [[lbl]]
        inp = self.sp(images=img, return_tensors="pt", **kw).to(DEV)
        inp = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) and k == "pixel_values" else v) for k, v in inp.items()}
        out = self.sm(**inp, multimask_output=True)
        masks = self.sp.post_process_masks(out.pred_masks.float().cpu(), inp["original_sizes"].cpu())[0]
        ious = out.iou_scores.float().cpu().numpy().reshape(-1)
        m = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).numpy()
        if return_all:
            return [x > 0.5 for x in m], [float(v) for v in ious]
        k = int(np.argmax(ious))
        return m[k] > 0.5, float(ious[k])


def largest_cc(m: np.ndarray) -> tuple[np.ndarray, int]:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m.astype(bool), 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k, n - 1


def fill_holes(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8) * 255
    h, w = m8.shape
    ff = m8.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    return (m8 | cv2.bitwise_not(ff)) > 0


def bbox_of(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def subpixel_contour(gray: np.ndarray, mask: np.ndarray, search: float = 4.0, step: float = 0.25):
    """Refine the mask boundary along local normals to the max of the gradient magnitude (parabolic peak)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    n = len(c)
    k = 5
    tang = np.roll(c, -k, 0) - np.roll(c, k, 0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    ts = np.arange(-search, search + 1e-6, step, dtype=np.float32)
    P = c[:, None, :] + ts[None, :, None] * nrm[:, None, :]
    mx = cv2.remap(mag, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    j = np.argmax(mx, 1)
    jj = np.clip(j, 1, len(ts) - 2)
    a, b, cc = mx[np.arange(n), jj - 1], mx[np.arange(n), jj], mx[np.arange(n), jj + 1]
    den = a - 2 * b + cc
    off = np.where(np.abs(den) > 1e-6, 0.5 * (a - cc) / den, 0.0)
    t = ts[jj] + np.clip(off, -1, 1) * step
    ok = (j > 0) & (j < len(ts) - 1)
    t = np.where(ok, t, 0.0)
    refined = c + t[:, None] * nrm
    return c, refined.astype(np.float32)


def box_iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def cross_check(sam_m: np.ndarray, bn_m: np.ndarray):
    """SAM mask looked like backdrop. Two independent models agreeing (IoU >= 0.8) means the object really fills the
    frame (close-ups): keep SAM. Otherwise take BiRefNet unless it is itself the whole frame / a rectangle -> None (fail)."""
    if sam_m.any() and mask_iou(sam_m, bn_m) >= 0.8:
        return sam_m, "agree_birefnet"
    if (not bn_m.any()) or bn_m.mean() > 0.95:
        return None, "birefnet_backdrop"
    x0, y0, x1, y1 = bbox_of(bn_m)
    if bn_m.sum() / max(1, (x1 - x0) * (y1 - y0)) >= 0.97:
        return None, "birefnet_rectangle"
    return bn_m, "birefnet"


def suspect_reasons(m: np.ndarray, props, tile_score: float, bn: np.ndarray) -> list[str]:
    """Typical v1 failure signatures on public images (E-27). Empty list = mask looks like an object.
    A large, border-touching mask that BiRefNet independently agrees with (IoU >= 0.8) is a close-up, not backdrop."""
    r = []
    if not m.any():
        return ["empty"]
    edges = sum([m[0, :].any(), m[-1, :].any(), m[:, 0].any(), m[:, -1].any()])
    if (m.mean() > 0.9 or (edges >= 3 and m.mean() > 0.5)) and mask_iou(m, bn) < 0.8:
        r.append("backdrop")
    x0, y0, x1, y1 = bbox_of(m)
    if m.sum() / max(1, (x1 - x0) * (y1 - y0)) >= 0.97:
        r.append("rectangle")
    n = m.sum()
    for b, sc, lab in props:
        X0, Y0, X1, Y1 = [int(round(v)) for v in b]
        if sc > tile_score and m[max(0, Y0):Y1, max(0, X0):X1].sum() > 0.8 * n:
            r.append(f"inside_prop:{lab}")
            break
    return r


def square_crop(im: Image.Image, box, pad_frac=0.05, out=1024):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad_frac)
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    return (sx0, sy0, sx0 + s, sy0 + s)


def process(models: Models, iid: str, path: str, modality: str, out_dir: str) -> dict:
    t0 = time.time()
    row = {"image_id": iid, "ok": 0}
    im = load_rgb(path)
    W0, H0 = im.size
    row.update(orig_w=W0, orig_h=H0)
    s = min(1.0, 2048 / max(W0, H0))
    work = im.resize((round(W0 * s), round(H0 * s)), Image.LANCZOS) if s < 1 else im
    row["work_scale"] = round(s, 6)
    Ww, Hw = work.size
    # ---- pass 1: concept prompt -> box -> coarse mask
    dets = models.detect(work, RUB_PROMPT if modality == "rubbing" else TILE_PROMPT, thr=0.2)
    dets = [d for d in dets if (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) > 0.02 * Ww * Hw]
    if not dets:  # second chance with generic object words (3/4 views, heads, gradient backgrounds)
        dets = models.detect(work, "terracotta object. ceramic sculpture. tile.", thr=0.15)
        dets = [d for d in dets if 0.02 * Ww * Hw < (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) < 0.95 * Ww * Hw]
        if dets:
            row["det_label"] = "generic:"
    row["n_det"] = len(dets)
    props = [d for d in models.detect(work, PROP_PROMPT, thr=0.3)
             if (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) > 0.005 * Ww * Hw]
    row["n_prop"] = len(props)
    keep = [d for d in dets if not any(box_iou(d[0], p[0]) > 0.6 and p[1] > d[1] for p in props)]
    row["skipped_prop_boxes"] = len(dets) - len(keep)
    dets = keep

    def swallowed(m):  # sanity check: mask = backdrop (covers most of the frame, or >50 % and touching >= 3 borders)
        edges = sum([m[0, :].any(), m[-1, :].any(), m[:, 0].any(), m[:, -1].any()])
        return (not m.any()) or m.mean() > 0.9 or (edges >= 3 and m.mean() > 0.5)
    row["rescue"] = ""
    m1, iou1 = None, 0.0
    if dets:
        # v3.2 shape prior from the agreed definition (PROGRESS E-28): a 瓦当 is round. In paired photos the 軒平瓦 got
        # the top score, and seen at an angle its box is only 1.4-2.1 long/short, while the 瓦当 box is ~1.0 (review W0364,
        # cloud/debug_dets.py). Drop whole-frame boxes; if the top box is not round (> 1.3) and a clearly round box
        # (<= 1.15) with >= 60 % of its score lies mostly OUTSIDE it (< 70 % contained; parts of the same object stay),
        # take the round box. A single box, or no round alternative, is never affected.
        def asp(d):
            w, h = d[0][2] - d[0][0], d[0][3] - d[0][1]
            return max(w, h) / max(1e-6, min(w, h))

        def contained(a, b):  # share of box a inside box b
            ix = max(0.0, min(a[0][2], b[0][2]) - max(a[0][0], b[0][0]))
            iy = max(0.0, min(a[0][3], b[0][3]) - max(a[0][1], b[0][1]))
            return ix * iy / max(1e-6, (a[0][2] - a[0][0]) * (a[0][3] - a[0][1]))
        pool = [d for d in dets if (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) <= 0.9 * Ww * Hw] or dets
        best = max(pool, key=lambda d: d[1])
        rounds = [d for d in pool if d is not best and asp(d) <= 1.15 and d[1] >= 0.6 * best[1] and contained(d, best) < 0.7]
        pick = max(rounds, key=lambda d: d[1]) if (asp(best) > 1.3 and rounds) else best
        box, sc, lab = pick
        row["shape_prior"] = int(pick is not best)
        row.update(det_label=row.get("det_label", "") + lab, det_score=round(sc, 3))
        m1, iou1 = models.segment(work, box)
        m1, _ = largest_cc(m1)
    if m1 is None or swallowed(m1):  # no box, or the mask looks like backdrop -> cross-check with a different model
        bn1 = largest_cc(models.salient(work))[0]
        m1, how = cross_check(m1 if m1 is not None else np.zeros_like(bn1), bn1)
        row["rescue"] = "p1:" + how + ("" if dets else "_nodet")
        if m1 is None:
            raise RuntimeError("pass 1: backdrop, BiRefNet cross-check failed (" + how + ")")
    row["sam_iou_p1"] = round(iou1, 3)
    bb = bbox_of(m1)
    if bb is None:
        raise RuntimeError("empty coarse mask")
    # ---- pass 2: re-segment inside bbox+10% at higher effective resolution (from the original pixels)
    bx0, by0, bx1, by1 = [v / s for v in bb]
    bw, bh = bx1 - bx0, by1 - by0
    cx0, cy0 = max(0, int(bx0 - 0.1 * bw)), max(0, int(by0 - 0.1 * bh))
    cx1, cy1 = min(W0, int(np.ceil(bx1 + 0.1 * bw))), min(H0, int(np.ceil(by1 + 0.1 * bh)))
    crop = im.crop((cx0, cy0, cx1, cy1))
    cs = min(1.0, 2048 / max(crop.size))
    if cs < 1:
        crop = crop.resize((round(crop.size[0] * cs), round(crop.size[1] * cs)), Image.LANCZOS)
    cw, ch = crop.size
    tb = [(bx0 - cx0) * cs, (by0 - cy0) * cs, (bx1 - cx0) * cs, (by1 - cy0) * cs]
    m1c = cv2.resize(m1.astype(np.uint8), None, fx=1, fy=1)
    # positive point: centroid of the coarse mask (mapped), if inside; negatives: crop corners outside the coarse mask
    ys, xs = np.nonzero(m1)
    pc = ((xs.mean() / s - cx0) * cs, (ys.mean() / s - cy0) * cs)
    neg = [(3, 3), (cw - 4, 3), (3, ch - 4), (cw - 4, ch - 4)]
    m1_in_crop = cv2.resize(m1.astype(np.uint8), (Ww, Hw))
    def inside_coarse(p):
        X, Y = int((p[0] / cs + cx0) * s), int((p[1] / cs + cy0) * s)
        return 0 <= X < Ww and 0 <= Y < Hw and m1_in_crop[Y, X] > 0
    neg = [p for p in neg if not inside_coarse(p)]
    pos = [pc] if inside_coarse(pc) else []
    bn2 = fill_holes(largest_cc(models.salient(crop))[0])  # BiRefNet on every crop: granularity, rescue, agreement
    cands, scores = models.segment(crop, tb, pos=pos, neg=neg, return_all=True)
    k = int(np.argmax(scores))
    # v3.1: SAM returns three granularities; v1/v3 took SAM's own top score, which on antefixes was often a part (hair,
    # outer ring, bottom band; review W0360). Pick the candidate that best matches BiRefNet's whole-object mask, unless
    # BiRefNet itself looks like backdrop / a rectangle or matches nothing (then keep SAM's top score).
    bn_ok = bn2.any() and bn2.mean() <= 0.9 and cross_check(np.zeros_like(bn2), bn2)[0] is not None
    if bn_ok:
        ious_bn = [mask_iou(fill_holes(largest_cc(c)[0]), bn2) for c in cands]
        if max(ious_bn) >= 0.5:
            k = int(np.argmax(ious_bn))
    row["sam_pick"] = "bn" if k != int(np.argmax(scores)) else "score"
    m2, iou2 = cands[k], scores[k]
    m2, ncc = largest_cc(m2)
    m2 = fill_holes(m2)
    edges2 = sum([m2[0, :].any(), m2[-1, :].any(), m2[:, 0].any(), m2[:, -1].any()])
    if edges2 >= 3 and m2.mean() > 0.75:  # pass 2 looks like the whole crop -> cross-check (no coarse-mask fallback)
        m2, how = cross_check(m2, bn2)
        row["rescue"] = (row["rescue"] + "+" if row["rescue"] else "") + "p2:" + how
        if m2 is None:
            raise RuntimeError("pass 2: whole crop, BiRefNet cross-check failed (" + how + ")")
    row.update(sam_iou_p2=round(iou2, 3), n_components=ncc)
    # ---- occluders (mount, clamp, rod): kept as their own class, removed from tile_front
    occ = np.zeros_like(m2)
    tile_area = m2.sum()
    for b, sc, lab in models.detect(crop, OCC_PROMPT, thr=0.35):
        barea = (b[2] - b[0]) * (b[3] - b[1])
        if barea > 0.5 * tile_area or barea < 50:
            continue
        mo, _ = models.segment(crop, b)
        if mo.sum() > 0.3 * tile_area:
            continue
        near = cv2.dilate(m2.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        occ |= mo & near
    tile = m2 & ~occ
    props2 = [d for d in models.detect(crop, PROP_PROMPT, thr=0.3) if (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) > 0.005 * cw * ch]
    sus = suspect_reasons(tile, props2, float(row.get("det_score") or 0), bn2)
    if sus and "birefnet" not in row["rescue"]:  # first suspicious result -> one rescue with the other model
        m2 = bn2
        occ &= m2
        tile = m2 & ~occ
        row["rescue"] = (row["rescue"] + "+" if row["rescue"] else "") + "birefnet_check:" + "|".join(sus)
        sus = suspect_reasons(tile, props2, float(row.get("det_score") or 0), bn2)
    if sus:  # never accept a suspicious mask silently (E-27)
        raise RuntimeError("suspect mask after rescue: " + "|".join(sus))
    row["bn_iou"] = round(mask_iou(tile | occ, bn2), 3)
    row["suspect"] = "disagree_birefnet" if row["bn_iou"] < 0.5 else ""  # kept, but first in line for the review
    tile_area = m2.sum()  # occluder fraction relative to the whole front, as in v1
    row.update(occluder_px=int(occ.sum()), occluder_frac=round(float(occ.sum()) / max(1, tile_area), 4))
    row["mask_frac"] = round(float(tile.sum()) / (cw * ch), 4)
    # ---- geometry helpers
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    raw_c, sub_c = subpixel_contour(gray, tile)
    per = cv2.arcLength(raw_c.reshape(-1, 1, 2), True) if len(raw_c) else 0
    row["circularity"] = round(float(4 * np.pi * tile.sum() / (per ** 2)) if per else 0, 4)
    tb2 = bbox_of(tile)
    row["touches_border"] = int(tb2[0] <= 1 or tb2[1] <= 1 or tb2[2] >= cw - 1 or tb2[3] >= ch - 1) if tb2 else 1
    os.makedirs(os.path.join(out_dir, "seg"), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "seg", f"{iid}.npz"), tile=tile, occ=occ, raw_contour=raw_c,
                        sub_contour=sub_c, crop_box=np.array([cx0, cy0, cx1, cy1]), crop_scale=cs,
                        orig_size=np.array([W0, H0]))
    crop.save(os.path.join(out_dir, "seg", f"{iid}_p2.jpg"), quality=92)
    # ---- std outputs: square crop around tile bbox (+5%), 1024 px, same frame for mask
    ob = [tb2[0] / cs + cx0, tb2[1] / cs + cy0, tb2[2] / cs + cx0, tb2[3] / cs + cy0]
    sq = square_crop(im, ob)
    side = sq[2] - sq[0]
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(im.crop((max(0, sq[0]), max(0, sq[1]), min(W0, sq[2]), min(H0, sq[3]))),
                 (max(0, -sq[0]), max(0, -sq[1])))
    std_img = canvas.resize((1024, 1024), Image.LANCZOS)
    full = np.zeros((side, side), np.uint8)
    lab_crop = np.where(tile, 255, np.where(occ, 128, 0)).astype(np.uint8)
    lab_orig = cv2.resize(lab_crop, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_NEAREST)
    ox, oy = cx0 - sq[0], cy0 - sq[1]
    x0d, y0d = max(0, ox), max(0, oy)
    x0s, y0s = x0d - ox, y0d - oy
    wv, hv = min(side - x0d, lab_orig.shape[1] - x0s), min(side - y0d, lab_orig.shape[0] - y0s)
    full[y0d:y0d + hv, x0d:x0d + wv] = lab_orig[y0s:y0s + hv, x0s:x0s + wv]
    std_mask = cv2.resize(full, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    os.makedirs(os.path.join(out_dir, "std"), exist_ok=True)
    std_img.save(os.path.join(out_dir, "std", f"{iid}_crop.jpg"), quality=92, icc_profile=ImageCms.ImageCmsProfile(_srgb).tobytes())
    Image.fromarray(std_mask).save(os.path.join(out_dir, "std", f"{iid}_mask.png"), optimize=True)
    row.update(crop_x0=sq[0], crop_y0=sq[1], crop_x1=sq[2], crop_y1=sq[3],
               crop_file=f"std/{iid}_crop.jpg", mask_file=f"std/{iid}_mask.png", ok=1, error="",
               sec=round(time.time() - t0, 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-test", action="store_true", help="score candidate prompt words on the manifest and exit")
    ap.add_argument("--shard", default="", help="i/n: every n-th row from i (parallel processes on one GPU)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]
    idx_name = "seg_index.csv"
    if a.shard:
        si, sn = map(int, a.shard.split("/"))
        rows = rows[si::sn]
        idx_name = f"seg_index_shard{si}of{sn}.csv"
    models = Models()
    if a.prompt_test:
        words = ["roof tile end.", "eaves tile.", "antefix.", "round tile.", "ceramic disk.", "roof tile."]
        res = {w: [] for w in words}
        for r in rows:
            im = load_rgb(r["path"])
            im.thumbnail((1024, 1024))
            for w in words:
                d = models.detect(im, w, thr=0.05)
                res[w].append(max([x[1] for x in d], default=0.0))
        print("PROMPT_TEST", json.dumps({w: {"mean_max": round(float(np.mean(v)), 3), "frac>0.3": round(float(np.mean(np.array(v) > 0.3)), 3)} for w, v in res.items()}))
        return
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, idx_name)
    done = set()
    if os.path.exists(idx_path):
        done = {r["image_id"] for r in csv.DictReader(open(idx_path, encoding="utf-8")) if r["ok"] == "1"}
    new = not os.path.exists(idx_path)
    f = open(idx_path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t0 = time.time()
    for i, r in enumerate(rows):
        if r["image_id"] in done:
            continue
        try:
            row = process(models, r["image_id"], r["path"], r.get("modality", "photo"), a.out)
        except Exception as e:
            row = {"image_id": r["image_id"], "ok": 0, "error": f"{type(e).__name__}: {e}"[:300]}
            traceback.print_exc()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        f.flush()
        if i % 20 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    f.close()
    print("SEG_DONE", flush=True)


if __name__ == "__main__":
    main()
