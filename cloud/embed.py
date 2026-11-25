"""Stage 2 embeddings v2 (cloud): Perception Encoder PE-Core G/14-448 image embeddings (timm), L2-normalised.

Model: timm/vit_pe_core_gigantic_patch14_448.fb (Apache-2.0, Meta 2025). Replaces DINOv2-L (2023): DINOv3 is gated with a
non-permissive licence, so PE-Core-G is the newest image encoder that satisfies 方案 1.6.
Usage: venv/bin/python embed.py manifest.csv OUT.npz
"""
import csv
import sys

import numpy as np
import timm
import torch

from segment import load_rgb

MODEL = "hf-hub:timm/vit_pe_core_gigantic_patch14_448.fb"
man, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(man, encoding="utf-8")))
m = timm.create_model(MODEL, pretrained=True, num_classes=0).to("cuda").eval()
cfg = timm.data.resolve_data_config({}, model=m)
tf = timm.data.create_transform(**cfg)
ids, embs = [], []
B = 32
for i in range(0, len(rows), B):
    batch = rows[i:i + B]
    x = torch.stack([tf(load_rgb(r["path"])) for r in batch]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        e = m(x).float().cpu().numpy()
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    embs.append(e)
    ids += [r["image_id"] for r in batch]
    if i % 320 == 0:
        print(i, flush=True)
np.savez_compressed(out, ids=np.array(ids), emb=np.concatenate(embs), model=MODEL)
print("EMBED_DONE", len(ids), embs[0].shape)
