"""Stage 4 v2 prompt selection (方案 Stage 4: "eaves tile" / "roof tile end" / "antefix"，试几个词取最稳的).
Public museum images only. Judged by hit rate + visual contact sheets; no reference masks from the discarded v1.

Model input: native aspect ratio, long side 1024 (model default, README). One control arm repeats "roof tile end"
with the old letterbox-to-1024x1024 input, to see whether the letterbox explains the low hit rate (W0343).

Per (image, query): n instances, area of the largest, seconds; the largest mask is saved at <=512 px for the sheets.
Usage: venv/bin/python prompt_explore.py sample.csv OUT_DIR --shard i/n
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from segment import MAX_NEW_TOKENS, Falcon, load_rgb

PLAN_WORDS = ["roof tile end", "eaves tile", "antefix"]          # 方案原文
NEAR_WORDS = ["tile end", "circular roof tile end", "round tile"]  # 近义变体
RUBBING_WORDS = ["ink rubbing"]
LETTERBOX_CONTROL = "roof tile end"


@torch.no_grad()
def native_masks(f: Falcon, img, q):
    preds = f.m.generate(img, q, compile=False, max_new_tokens=MAX_NEW_TOKENS, min_dimension=256, max_dimension=1024)[0]
    out = []
    for p in preds:
        rle = p["mask_rle"]
        m = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("utf-8")}).astype(bool)
        if m.shape != (img.size[1], img.size[0]):
            m = cv2.resize(m.astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST) > 0
        if m.any():
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    rows = list(csv.DictReader(open(a.sample, encoding="utf-8")))[si::sn]
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f"res_{si}of{sn}.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path, encoding="utf-8"):
            d = json.loads(line)
            done.add((d["image_id"], d["arm"]))
    fo = open(res_path, "a", encoding="utf-8")
    f = Falcon()
    t0 = time.time()
    for n, r in enumerate(rows):
        im = load_rgb(r["path"])
        W, H = im.size
        s = min(1.0, 512 / max(W, H))
        dw, dh = max(1, round(W * s)), max(1, round(H * s))
        arms = [(q, "native") for q in PLAN_WORDS + NEAR_WORDS]
        if r["modality"] != "photo":
            arms += [(q, "native") for q in RUBBING_WORDS]
        arms.append((LETTERBOX_CONTROL, "letterbox"))
        masks = {}
        for q, mode in arms:
            arm = f"{mode}:{q}"
            if (r["image_id"], arm) in done:
                continue
            torch.cuda.synchronize()
            t = time.time()
            inst = native_masks(f, im, q) if mode == "native" else f.masks(im, q)
            torch.cuda.synchronize()
            rec = {"image_id": r["image_id"], "arm": arm, "query": q, "mode": mode, "n": len(inst),
                   "max_area": max((float(m.mean()) for m in inst), default=0.0), "sec": round(time.time() - t, 2),
                   "work_w": W, "work_h": H, "source": r["source"], "object_type": r["object_type"], "modality": r["modality"]}
            if inst:
                big = max(inst, key=lambda m: m.sum())
                masks[arm] = cv2.resize(big.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_AREA) > 0
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
        if masks:
            np.savez_compressed(os.path.join(a.out, f"{r['image_id']}.npz"), shape=np.array([dh, dw]),
                                **{k.replace(" ", "_").replace(":", "__"): np.packbits(m) for k, m in masks.items()})
        print(f"shard {si} {n + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("EXPLORE_DONE", flush=True)


if __name__ == "__main__":
    main()
