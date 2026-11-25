"""Stage 0 smoke test (v2): Falcon-Perception and MoGe-2 each run on one public image.
Usage (cloud): venv/bin/python smoke.py <image.jpg>
"""
import json
import sys
import time

import numpy as np
import torch

from segment import Falcon, load_rgb

img = load_rgb(sys.argv[1])
rep = {}
t = time.time()
f = Falcon()
ms = f.masks(img, "roof tile end")
rep["falcon"] = {"n_instances": len(ms), "area_frac": [round(float(m.mean()), 4) for m in ms], "sec": round(time.time() - t, 1)}
print("FALCON", rep["falcon"], flush=True)
t = time.time()
from moge.model.v3 import MoGeModel
mm = MoGeModel.from_pretrained("Ruicheng/moge-3-vitg").to("cuda").eval()
small = img.copy()
small.thumbnail((1024, 1024))
with torch.no_grad():
    o = mm.infer(torch.tensor(np.asarray(small) / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1))
rep["moge2"] = {"points": list(o["points"].shape), "sec": round(time.time() - t, 1)}
print("MOGE2", rep["moge2"])
print("SMOKE_JSON", json.dumps(rep))
