#!/bin/bash
# v3 GPU batch (PROGRESS E-26/E-27): Stage 4 = GDINO + SAM 2.1 without image-specific heuristics, BiRefNet cross-check;
# Stage 5 = MoGe-3. Stage 2 embeddings are NOT rerun (v1 DINOv2 queue kept). Inputs: work/manifest_v3.csv. Outputs: work/out_v3/
cd /workspace/wadang
export HF_HOME=/workspace/wadang/models/hf HF_HUB_OFFLINE=1
M=work/manifest_v3.csv
O=work/out_v3
mkdir -p $O logs
date -u +%FT%TZ > $O/started.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $O/gpu.txt
venv/bin/pip freeze > $O/pip_freeze.txt
echo "== segment v3 (4 shards x 6 threads = the 24-CPU quota)"
export NTHREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for i in 0 1 2 3; do venv/bin/python segment.py $M $O --shard $i/4 > logs/seg_v3_$i.log 2>&1 & done
wait
venv/bin/python - <<PY
import csv, glob
rows = []
for p in sorted(glob.glob("$O/seg_index_shard*of4.csv")):
    rows += list(csv.DictReader(open(p, encoding="utf-8")))
with open("$O/seg_index.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("merged", len(rows), "ok", sum(r["ok"] == "1" for r in rows))
PY
export NTHREADS=24 OMP_NUM_THREADS=24
echo "== geometry (MoGe-3)";  venv/bin/python geometry.py $O $M $O > logs/geometry_v3.log 2>&1; tail -1 logs/geometry_v3.log
echo "== synthetic";          venv/bin/python geometry.py $O $M $O --synthetic 30 > logs/synthetic_v3.log 2>&1; tail -1 logs/synthetic_v3.log
date -u +%FT%TZ > $O/finished.txt
echo BATCH_V3_DONE
