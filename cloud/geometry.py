"""Stage 5 geometry (cloud). Rim ellipse (RANSAC, occluder/border segments excluded, inner rim preferred),
MoGe-2 plane normal as an independent tilt estimate, completeness, QA overlays; plus synthetic tilt validation.

Usage:
  venv/bin/python geometry.py SEG_DIR manifest.csv OUT_DIR                 # -> OUT_DIR/rim.csv, OUT_DIR/overlay/*.jpg
  venv/bin/python geometry.py SEG_DIR manifest.csv OUT_DIR --no-moge
  venv/bin/python geometry.py SEG_DIR manifest.csv OUT_DIR --synthetic N    # -> OUT_DIR/tilt_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np

RNG = np.random.default_rng(20261108)
RIM_FIELDS = ["image_id", "ok", "error", "shape", "rim_edge_used", "rim_arc_deg", "rim_arc_deg_outer", "inlier_frac",
              "cx_px", "cy_px", "a_px", "b_px", "angle_deg", "a_in_px", "b_in_px", "angle_in_deg", "tilt_deg_ellipse", "tilt_deg_normal",
              "normal_inlier_frac", "tilt_deg", "tilt_source", "tilt_disagree", "inner_rejected", "residual_px", "completeness_ratio",
              "completeness_method", "n_points", "n_excluded_occ", "n_excluded_border"]


# ------------------------------------------------------------------------------------------ ellipse utils
def ell_params(e):
    (cx, cy), (w, h), ang = e
    a, b = max(w, h) / 2, min(w, h) / 2
    theta = math.radians(ang) if w >= h else math.radians(ang + 90)  # direction of the major axis
    return cx, cy, a, b, theta


def ell_dist(P, e):
    cx, cy, a, b, th = ell_params(e)
    c, s = math.cos(th), math.sin(th)
    x, y = P[:, 0] - cx, P[:, 1] - cy
    u, v = c * x + s * y, -s * x + c * y
    r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return np.abs(r - 1) * math.sqrt(a * b), np.arctan2(v / b, u / a)


def fit_direct(P):
    if len(P) < 6:
        return None
    try:
        e = cv2.fitEllipseDirect(P.reshape(-1, 1, 2).astype(np.float32))
    except cv2.error:
        return None
    (_, _), (w, h), _ = e
    if not (np.isfinite(w) and np.isfinite(h)) or min(w, h) <= 2:
        return None
    return e


def ransac_ellipse(P, thr, iters=600, min_axis=10):
    best, best_in = None, None
    n = len(P)
    if n < 12:
        return None, None
    for _ in range(iters):
        idx = RNG.choice(n, 6, replace=False)
        e = fit_direct(P[idx])
        if e is None:
            continue
        _, _, a, b, _ = ell_params(e)
        if b < min_axis or a > 1e5:
            continue
        d, _ = ell_dist(P, e)
        inl = d < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = e, inl
    if best is None:
        return None, None
    for _ in range(3):
        e = fit_direct(P[best_in])
        if e is None:
            break
        d, _ = ell_dist(P, e)
        best, best_in = e, d < thr
    return best, best_in


def arc_coverage_deg(phis, bin_deg=2.0, gap_fill_deg=4.0):
    if len(phis) == 0:
        return 0.0
    nb = int(round(360 / bin_deg))
    b = np.zeros(nb, bool)
    b[((np.degrees(phis) % 360) / bin_deg).astype(int) % nb] = True
    # fill small gaps (circular)
    k = int(gap_fill_deg / bin_deg)
    bb = b.copy()
    idx = np.nonzero(b)[0]
    if len(idx) == 0:
        return 0.0
    for i in range(nb):
        if not b[i]:
            l = any(b[(i - j) % nb] for j in range(1, k + 1))
            r = any(b[(i + j) % nb] for j in range(1, k + 1))
            bb[i] = l and r
    return float(bb.sum() * bin_deg)


def fit_rim(P, shape_hw, occ=None, border=2):
    """Outer rim ellipse from sub-pixel contour points P (N,2) in crop pixels."""
    H, W = shape_hw
    keep = np.ones(len(P), bool)
    n_occ = 0
    if occ is not None and occ.any():
        dt = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3)
        xi, yi = np.clip(P[:, 0].round().astype(int), 0, W - 1), np.clip(P[:, 1].round().astype(int), 0, H - 1)
        near = dt[yi, xi] < 6
        n_occ = int(near.sum())
        keep &= ~near
    on_border = (P[:, 0] <= border) | (P[:, 1] <= border) | (P[:, 0] >= W - 1 - border) | (P[:, 1] >= H - 1 - border)
    keep &= ~on_border
    Q = P[keep]
    scale = max(H, W)
    thr = max(1.5, 0.004 * scale)
    e, inl = ransac_ellipse(Q, thr, min_axis=0.08 * scale)
    if e is None:
        return None
    d, phis = ell_dist(Q, e)
    arc = arc_coverage_deg(phis[inl])
    return {"e": e, "inl_frac": float(inl.mean()), "arc": arc, "resid": float(np.sqrt(np.mean(d[inl] ** 2))),
            "n": int(len(P)), "n_occ": n_occ, "n_border": int(on_border.sum())}


def inner_rim(gray, tile, e_out):
    """Look for an edge concentric with the outer rim at normalised radius 0.62-0.95 (the rim's inner edge, on the
    tile's front plane). Returns ellipse or None."""
    cx, cy, a, b, th = ell_params(e_out)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    phis = np.radians(np.arange(0, 360, 2.0))
    rhos = np.arange(0.55, 0.985, 0.005)
    c, s = math.cos(th), math.sin(th)
    U = rhos[None, :] * a * np.cos(phis)[:, None]
    V = rhos[None, :] * b * np.sin(phis)[:, None]
    X = (cx + c * U - s * V).astype(np.float32)
    Y = (cy + s * U + c * V).astype(np.float32)
    prof = cv2.remap(g, X, Y, cv2.INTER_LINEAR)
    inside = cv2.remap(tile.astype(np.float32), X, Y, cv2.INTER_NEAREST) > 0.5
    dr = np.abs(np.diff(prof, axis=1))
    dr[~(inside[:, 1:] & inside[:, :-1])] = np.nan
    valid_rows = np.mean(np.isfinite(dr), 1) > 0.9
    if valid_rows.sum() < 60:
        return None
    mean_prof = np.nanmean(dr[valid_rows], 0)
    r_mid = (rhos[1:] + rhos[:-1]) / 2
    band = (r_mid >= 0.62) & (r_mid <= 0.95)
    med = np.nanmedian(mean_prof[(r_mid >= 0.55) & (r_mid <= 0.97)])
    cand = [i for i in np.nonzero(band)[0]
            if mean_prof[i] >= 1.8 * med and mean_prof[i] == np.nanmax(mean_prof[max(0, i - 3): i + 4])]
    if not cand:
        return None
    i_star = max(cand)  # outermost strong concentric edge = rim inner edge
    pts = []
    for k in np.nonzero(valid_rows)[0]:
        lo, hi = max(0, i_star - 8), min(len(r_mid), i_star + 9)
        seg = dr[k, lo:hi]
        if not np.isfinite(seg).any():
            continue
        j = lo + int(np.nanargmax(seg))
        if dr[k, j] >= 1.3 * med:
            pts.append((float((X[k, j] + X[k, j + 1]) / 2), float((Y[k, j] + Y[k, j + 1]) / 2)))
    if len(pts) < 40:
        return None
    P = np.array(pts, np.float32)
    e, inl = ransac_ellipse(P, max(1.5, 0.004 * 2 * a), min_axis=0.3 * b)
    if e is None:
        return None
    _, phi_in = ell_dist(P[inl], e)
    arc = arc_coverage_deg(phi_in)
    cxi, cyi, ai, bi, _ = ell_params(e)
    # The inner rim edge lies on the front plane, so under real tilt it is MORE elliptical than the outer silhouette
    # (which also contains the side wall): allow bi/ai below b/a by up to 0.15, but not above it by more than 0.03.
    if arc < 240 or math.hypot(cxi - cx, cyi - cy) > 0.03 * a or not (b / a - 0.15 <= bi / ai <= b / a + 0.03):
        return None
    return {"e": e, "arc": arc, "inl_frac": float(inl.mean())}


# ------------------------------------------------------------------------------------------ inner-ring self-test
def _affine_ellipse(e, M):
    """Image of ellipse e under 2x3 affine M (exact: sample, transform, refit)."""
    cx, cy, a, b, th = ell_params(e)
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    c, s = math.cos(th), math.sin(th)
    P = np.stack([cx + a * np.cos(t) * c - b * np.sin(t) * s, cy + a * np.cos(t) * s + b * np.sin(t) * c], 1)
    Q = P @ M[:, :2].T + M[:, 2]
    return fit_direct(Q.astype(np.float32))


def inner_stable(gray, tile, e_out, inner, phi=15.0, tol_deg=1.0):
    """Accept the inner ring only if, after a known affine tilt of the frontal image (two directions), the detector
    finds the SAME ring again and the tilt it implies matches the known transform within tol_deg.
    Rationale: the first synthetic validation (WORKLOG) showed the detector can lock onto decorative rings
    (text-field borders, bead bands) and jump between them under tilt."""
    cx, cy, _, _, _ = ell_params(e_out)
    H, W = tile.shape
    k = math.cos(math.radians(phi))
    for psi in (0.0, 90.0):
        c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
        R = np.array([[c, -s], [s, c]])
        A2 = R @ np.diag([k, 1.0]) @ R.T
        t = np.array([cx, cy]) - A2 @ np.array([cx, cy])
        M = np.hstack([A2, t[:, None]])
        g2 = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR)
        t2 = cv2.warpAffine(tile.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        e_out2 = _affine_ellipse(e_out, M)
        e_exp = _affine_ellipse(inner["e"], M)
        if e_out2 is None or e_exp is None:
            return False
        got = inner_rim(g2, t2, e_out2)
        if got is None:
            return False
        _, _, ag, bg, _ = ell_params(got["e"])
        _, _, ae, be, _ = ell_params(e_exp)
        if abs(ag / ae - 1) > 0.02:  # a different ring
            return False
        if abs(math.degrees(math.acos(min(1.0, bg / ag))) - math.degrees(math.acos(min(1.0, be / ae)))) > tol_deg:
            return False
    return True


# ------------------------------------------------------------------------------------------ MoGe plane
class Moge:
    def __init__(self):
        import torch
        from moge.model.v3 import MoGeModel  # MoGe-3 (2026-09, MIT); the v1 run used MoGe-2
        self.torch = torch
        self.m = MoGeModel.from_pretrained("Ruicheng/moge-3-vitg").to("cuda").eval()

    def tilt(self, rgb, mask_fit):
        torch = self.torch
        h, w = rgb.shape[:2]
        s = min(1.0, 1024 / max(h, w))
        if s < 1:
            rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            mask_fit = cv2.resize(mask_fit.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        with torch.no_grad():
            # refine_steps=0: MoGe-3's optional sparse 3D refiner needs flex_gemm 1.0.0, whose Triton kernels do not compile
            # on Triton 3.9 ("Cannot bitcast pointer between different address spaces"); the plane normal used for tilt
            # comes from the dense backbone output (WORKLOG)
            o = self.m.infer(x, refine_steps=0)
        pts = o["points"].float().cpu().numpy()
        valid = o["mask"].cpu().numpy() > 0.5 if "mask" in o else np.ones(pts.shape[:2], bool)
        sel = mask_fit & valid & np.isfinite(pts).all(-1)
        X = pts[sel]
        if len(X) < 200:
            return None, None
        if len(X) > 20000:
            X = X[RNG.choice(len(X), 20000, replace=False)]
        depth = np.median(X[:, 2])
        thr = 0.004 * depth
        best_n, best_in = None, None
        for _ in range(300):
            p = X[RNG.choice(len(X), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            d = np.abs((X - p[0]) @ n)
            inl = d < thr
            if best_in is None or inl.sum() > best_in.sum():
                best_n, best_in = n, inl
        Y = X[best_in]
        cen = Y.mean(0)
        _, _, vt = np.linalg.svd(Y - cen, full_matrices=False)
        n = vt[2]
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        return tilt, float(best_in.mean())


# ------------------------------------------------------------------------------------------ per image
def analyse(seg_dir, iid, modality, moge=None, force_edge=None):
    z = np.load(os.path.join(seg_dir, "seg", f"{iid}.npz"))
    tile, occ, P = z["tile"], z["occ"], z["sub_contour"]
    img = cv2.imdecode(np.fromfile(os.path.join(seg_dir, "seg", f"{iid}_p2.jpg"), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = tile.shape
    row = {"image_id": iid, "ok": 1, "error": ""}
    fit = fit_rim(P, (H, W), occ)
    area = float(tile.sum())
    e_used, edge_used, tilt_e, inner = None, "", None, None
    if fit:
        cx, cy, a, b, th = ell_params(fit["e"])
        row.update(rim_arc_deg_outer=round(fit["arc"], 1), inlier_frac=round(fit["inl_frac"], 3),
                   cx_px=round(cx, 2), cy_px=round(cy, 2), a_px=round(a, 2), b_px=round(b, 2),
                   angle_deg=round(math.degrees(th) % 180, 2), residual_px=round(fit["resid"], 3),
                   n_points=fit["n"], n_excluded_occ=fit["n_occ"], n_excluded_border=fit["n_border"])
        round_ok = fit["arc"] >= 200 and fit["inl_frac"] >= 0.5 and b / a >= 0.5
        if round_ok:
            inner = None if force_edge == "outer" else inner_rim(gray, tile, fit["e"])
            if inner is not None and force_edge is None and not inner_stable(gray, tile, fit["e"], inner):
                row["inner_rejected"] = 1
                inner = None
            e_used = inner["e"] if inner else fit["e"]
            edge_used = "inner" if inner else "outer"
            _, _, ai, bi, thi = ell_params(e_used)
            tilt_e = math.degrees(math.acos(min(1.0, bi / ai)))
            row.update(shape="round", rim_edge_used=edge_used,
                       rim_arc_deg=round(inner["arc"] if inner else fit["arc"], 1),
                       a_in_px=round(ai, 2) if inner else "", b_in_px=round(bi, 2) if inner else "",
                       angle_in_deg=round(math.degrees(thi) % 180, 2) if inner else "",
                       tilt_deg_ellipse=round(tilt_e, 2),
                       completeness_ratio=round(min(1.0, area / (math.pi * a * b)), 4), completeness_method="area/ellipse")
        elif fit["arc"] >= 90 and fit["inl_frac"] >= 0.5:
            row.update(shape="round_fragment", rim_arc_deg=round(fit["arc"], 1),
                       completeness_ratio=round(fit["arc"] / 360, 4), completeness_method="rim_arc/360 (no valid ellipse)")
        else:
            row.update(shape="non_round", rim_arc_deg=round(fit["arc"], 1))
    else:
        row.update(shape="non_round")
    if modality in ("rubbing", "drawing"):
        row.update(tilt_deg=0, tilt_source="rubbing")
        return row, img, fit, inner
    tilt_n = None
    if moge is not None:
        m_fit = cv2.erode(tile.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if fit and row.get("shape") == "round":
            cx, cy, a, b, th = ell_params(fit["e"])
            ell = np.zeros_like(tile, np.uint8)
            cv2.ellipse(ell, ((cx, cy), (2 * a * 0.85, 2 * b * 0.85), math.degrees(th)), 1, -1)
            m_fit &= ell > 0
        tilt_n, nin = moge.tilt(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), m_fit)
        if tilt_n is not None:
            row.update(tilt_deg_normal=round(tilt_n, 2), normal_inlier_frac=round(nin, 3))
    if row.get("shape") == "non_round":
        # E-14: for non-circular objects the MoGe plane is the dominant VISIBLE surface (e.g. the side of an antefix
        # in a profile photo), not the decorated face -> not published as tilt_deg (kept as tilt_deg_normal only).
        row.update(tilt_deg="", tilt_source="none")
    elif tilt_e is not None and tilt_n is not None:
        if abs(tilt_e - tilt_n) <= 5:
            row.update(tilt_deg=round((tilt_e + tilt_n) / 2, 2), tilt_source="both", tilt_disagree=0)
        else:
            row.update(tilt_deg=round(tilt_e, 2), tilt_source="ellipse", tilt_disagree=1)
    elif tilt_e is not None:
        row.update(tilt_deg=round(tilt_e, 2), tilt_source="ellipse")
    elif tilt_n is not None:
        row.update(tilt_deg=round(tilt_n, 2), tilt_source="normal")
    else:
        row.update(tilt_deg="", tilt_source="none")
    return row, img, fit, inner


def overlay(img, tile, occ, fit, inner, path):
    o = img.copy()
    cs, _ = cv2.findContours(tile.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cs, -1, (0, 255, 0), max(2, o.shape[1] // 400))
    if occ.any():
        o[occ] = (0.5 * o[occ] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    if fit:
        cv2.ellipse(o, fit["e"], (255, 0, 255), max(2, o.shape[1] // 400))
    if inner:
        cv2.ellipse(o, inner["e"], (0, 255, 255), max(2, o.shape[1] // 400))
    s = 640 / max(o.shape[:2])
    o = cv2.resize(o, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", o, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(path)


# ------------------------------------------------------------------------------------------ synthetic validation
def synthetic(seg_dir, rows_ok, n, out_dir):
    """Affine-tilt complete frontal tiles by known angles; run the ellipse estimator; measure angle error and
    marker-spacing deviation after rectification."""
    cands = [r for r in rows_ok if r.get("shape") == "round" and float(r.get("rim_arc_deg_outer") or 0) >= 359
             and float(r["b_px"]) / float(r["a_px"]) >= 0.98]
    cands = cands[:max(n, 20)]
    res = []
    for r in cands:
        iid = r["image_id"]
        z = np.load(os.path.join(seg_dir, "seg", f"{iid}.npz"))
        tile = z["tile"].astype(np.uint8)
        img = cv2.imdecode(np.fromfile(os.path.join(seg_dir, "seg", f"{iid}_p2.jpg"), np.uint8), cv2.IMREAD_COLOR)
        cx, cy, a = float(r["cx_px"]), float(r["cy_px"]), float(r["a_px"])
        # circularise the frontal tile first with its own fitted outer ellipse, so that the test measures the
        # estimator (edge + fit) error and not the object's intrinsic ellipticity (reported separately).
        use_inner = r.get("rim_edge_used") == "inner"
        if use_inner:
            a0, b0, th0 = float(r["a_in_px"]), float(r["b_in_px"]), math.radians(float(r["angle_in_deg"]))
        else:
            a0, b0, th0 = a, float(r["b_px"]), math.radians(float(r["angle_deg"]))
        c0, s0 = math.cos(th0), math.sin(th0)
        R0 = np.array([[c0, -s0], [s0, c0]])
        C = R0 @ np.diag([1.0, a0 / b0]) @ R0.T
        tC = np.array([cx, cy]) - C @ np.array([cx, cy])
        H0, W0 = tile.shape
        tile = cv2.warpAffine(tile, np.hstack([C, tC[:, None]]), (W0, H0), flags=cv2.INTER_NEAREST)
        img = cv2.warpAffine(img, np.hstack([C, tC[:, None]]), (W0, H0), flags=cv2.INTER_LINEAR)
        ang = RNG.uniform(0, 2 * math.pi, 40)
        rad = 0.8 * a * np.sqrt(RNG.uniform(0, 1, 40))
        markers = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
        for phi in (10, 20, 30):
            for psi in (0, 90, 37):  # yaw (compress along x), pitch (along y), oblique
                k = math.cos(math.radians(phi))
                c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
                R = np.array([[c, -s], [s, c]])
                A2 = R @ np.diag([k, 1.0]) @ R.T
                t = np.array([cx, cy]) - A2 @ np.array([cx, cy])
                M = np.hstack([A2, t[:, None]])
                H, W = tile.shape
                wt = cv2.warpAffine(tile, M, (W, H), flags=cv2.INTER_NEAREST) > 0
                wi = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(wi, cv2.COLOR_BGR2GRAY)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from segment import subpixel_contour
                _, P = subpixel_contour(gray, wt)
                fit = fit_rim(P, (H, W), None)
                if fit is None:
                    res.append({"image_id": iid, "phi": phi, "psi": psi, "ok": 0})
                    continue
                inner = inner_rim(gray, wt, fit["e"]) if use_inner else None
                if use_inner and inner is None:
                    res.append({"image_id": iid, "phi": phi, "psi": psi, "ok": 0, "edge": "inner_lost"})
                    continue
                e = inner["e"] if inner else fit["e"]
                ecx, ecy, ea, eb, eth = ell_params(e)
                est = math.degrees(math.acos(min(1.0, eb / ea)))
                # rectify: stretch along fitted minor axis by a/b
                cc, ss = math.cos(eth), math.sin(eth)
                Rm = np.array([[cc, -ss], [ss, cc]])
                Rect = Rm @ np.diag([1.0, ea / eb]) @ Rm.T
                Pm = (A2 @ markers.T).T + t
                Pr = (Rect @ (Pm - [ecx, ecy]).T).T
                P0 = markers - [cx, cy]
                i, j = np.triu_indices(len(P0), 1)
                d0 = np.linalg.norm(P0[i] - P0[j], axis=1)
                d1 = np.linalg.norm(Pr[i] - Pr[j], axis=1)
                ratio = d1 / d0
                ratio /= np.median(ratio)  # global scale is normalised away (std diameter 1024)
                res.append({"image_id": iid, "phi": phi, "psi": psi, "ok": 1, "est": round(est, 3),
                            "err": round(est - phi, 3), "edge": "inner" if inner else "outer",
                            "spacing_dev_median": round(float(np.median(np.abs(ratio - 1))), 5),
                            "spacing_dev_p95": round(float(np.percentile(np.abs(ratio - 1), 95)), 5)})
    with open(os.path.join(out_dir, "tilt_validation.json"), "w") as f:
        json.dump({"n_images": len(cands), "intrinsic_b_over_a": [round(float(r["b_px"]) / float(r["a_px"]), 4) for r in cands],
                   "frontal_complete_all_b_over_a": [round(float(r["b_px"]) / float(r["a_px"]), 4) for r in rows_ok
                                                      if r.get("shape") == "round" and float(r.get("rim_arc_deg_outer") or 0) >= 359],
                   "trials": res}, f, indent=1)
    print("SYNTH_DONE", len(cands), len(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--no-moge", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)
    man = {r["image_id"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    seg_rows = [r for r in csv.DictReader(open(os.path.join(a.seg_dir, "seg_index.csv"), encoding="utf-8")) if r["ok"] == "1"]
    if a.synthetic:
        rows_ok = list(csv.DictReader(open(os.path.join(a.out, "rim.csv"), encoding="utf-8")))
        synthetic(a.seg_dir, [r for r in rows_ok if r["ok"] == "1"], a.synthetic, a.out)
        return
    moge = None if a.no_moge else Moge()
    with open(os.path.join(a.out, "rim.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RIM_FIELDS)
        w.writeheader()
        t0 = time.time()
        for i, sr in enumerate(seg_rows):
            iid = sr["image_id"]
            try:
                row, img, fit, inner = analyse(a.seg_dir, iid, man.get(iid, {}).get("modality", "photo"), moge)
                z = np.load(os.path.join(a.seg_dir, "seg", f"{iid}.npz"))
                overlay(img, z["tile"], z["occ"], fit, inner, os.path.join(a.out, "overlay", f"{iid}.jpg"))
            except Exception as e:
                row = {"image_id": iid, "ok": 0, "error": f"{type(e).__name__}: {e}"[:300]}
                traceback.print_exc()
            w.writerow({k: row.get(k, "") for k in RIM_FIELDS})
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(seg_rows)} {time.time() - t0:.0f}s", flush=True)
    print("GEOM_DONE", flush=True)


if __name__ == "__main__":
    main()
