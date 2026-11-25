"""Clean-environment check (方案 Stage 7 / 验收表): load all three configs with `datasets` only and run one filter each.
Run with a venv that has ONLY `datasets` + `pillow`:  venv-clean/Scripts/python load_check.py dist/hf
"""
import sys
import time

from datasets import load_dataset

repo = sys.argv[1]
out = []
t0 = time.time()
d = load_dataset(repo, "default", split="train")
cc0 = d.filter(lambda lic: lic in ("CC0-1.0", "Public-Domain"), input_columns="license")
out.append(f"default: {d.num_rows} rows; licence filter CC0/PD -> {cc0.num_rows}; first image size {d[0]['image'].size}")
f = load_dataset(repo, "faces", split="train")
silla = f.filter(lambda pc: pc in ("KR-UnifiedSilla", "KR-Silla"), input_columns="period_code")
out.append(f"faces: {f.num_rows} rows; Silla filter -> {silla.num_rows}; all has_face: {all(f['has_face'])}")
s = load_dataset(repo, "std", split="train")
lotus = s.filter(lambda m, t: m == "lotus" and t is not None and t <= 20, input_columns=["motif", "tilt_deg"])
ex = s[0]
out.append(f"std: {s.num_rows} rows; lotus & tilt<=20 -> {lotus.num_rows}; image {ex['image'].size} mask {ex['mask'].size} "
           f"mask values {sorted(set(ex['mask'].getdata()))[:5]}")
out.append(f"datasets {__import__('datasets').__version__}; {time.time() - t0:.0f} s")
print("\n".join(out))
