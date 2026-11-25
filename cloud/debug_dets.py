"""Print every Grounding DINO box (score, label, aspect, area share) for a few public images (v3.1 debugging)."""
import sys
from PIL import Image
from segment import Models, load_rgb, TILE_PROMPT
ids = sys.argv[1].split(",")
m = Models()
for i in ids:
    im = load_rgb(f"work/upload/images/{i}.jpg")
    s = min(1.0, 2048 / max(im.size))
    work = im.resize((round(im.size[0] * s), round(im.size[1] * s))) if s < 1 else im
    W, H = work.size
    dets = m.detect(work, TILE_PROMPT, thr=0.1)
    print("==", i, W, H)
    for b, sc, lab in sorted(dets, key=lambda d: -d[1])[:8]:
        w, h = b[2] - b[0], b[3] - b[1]
        print(f"   {sc:.3f} {lab[:18]:18s} aspect {max(w, h) / max(1e-6, min(w, h)):.2f} area {w * h / (W * H):.3f} box {[round(v) for v in b]}")
