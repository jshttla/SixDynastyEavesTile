"""Quick timing of Falcon-Perception calls at different canvas sizes (public images only)."""
import csv, sys, time
import torch
from PIL import Image
sys.argv = sys.argv[:1]
import segment
from segment import Falcon, load_rgb
f = Falcon()
rows = list(csv.DictReader(open("work/prompt_sample.csv")))[:4]
for side in (None, 1024):
    for compile_flag in (False, True):
        segment.COMPILE = compile_flag
        ts = []
        for r in rows:
            im = load_rgb(r["path"])
            if side:
                im = im.copy(); im.thumbnail((side, side))
            torch.cuda.synchronize(); t = time.time()
            ms = f.masks(im, "roof tile end")
            torch.cuda.synchronize(); ts.append(time.time() - t)
        print(f"canvas={'orig' if side is None else side} compile={compile_flag} per-call s: {[round(x,2) for x in ts]} n_inst_last={len(ms)}", flush=True)
