import csv, time
import torch
from segment import Falcon, load_rgb
f = Falcon()
rows = list(csv.DictReader(open("work/prompt_sample.csv")))[:16]
ims = []
for r in rows:
    im = load_rgb(r["path"]); im.thumbnail((1024, 1024)); ims.append(im)
def run(imgs, mnt, comp):
    torch.cuda.synchronize(); t = time.time()
    out = f.m.generate(imgs, ["roof tile end"] * len(imgs), max_new_tokens=mnt, compile=comp)
    torch.cuda.synchronize(); return time.time() - t, [len(o) for o in out]
for comp in (True, False):
    for mnt in (128,):
        for bs in (1, 4, 8, 16):
            try:
                run(ims[:bs], mnt, comp)
                t, n = run(ims[:bs], mnt, comp)
                print(f"compile={comp} mnt={mnt} bs={bs}: {t:.2f}s total, {t/bs:.2f}s/img, mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB inst={n}", flush=True)
            except torch.OutOfMemoryError:
                print(f"compile={comp} mnt={mnt} bs={bs}: OOM", flush=True); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
