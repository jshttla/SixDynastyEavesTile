#!/bin/bash
# v3 Stage 5 rerun only (segmentation outputs in work/out_v3 are kept): MoGe-3 without the sparse refiner
cd /workspace/wadang
export HF_HOME=/workspace/wadang/models/hf HF_HUB_OFFLINE=1 NTHREADS=24 OMP_NUM_THREADS=24
M=work/manifest_v3.csv
O=work/out_v3
rm -f $O/rim.csv $O/tilt_validation.json; rm -rf $O/overlay
echo "== geometry (MoGe-3, refine_steps=0)"; venv/bin/python geometry.py $O $M $O > logs/geometry_v3b.log 2>&1; tail -1 logs/geometry_v3b.log
echo "== synthetic"; venv/bin/python geometry.py $O $M $O --synthetic 30 > logs/synthetic_v3b.log 2>&1; tail -1 logs/synthetic_v3b.log
echo GEOM_V3B_DONE
