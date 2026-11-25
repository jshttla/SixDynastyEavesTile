#!/bin/bash
# v2 GPU batch for the public images. Inputs: work/upload/{images,manifest.csv}. Outputs: work/out/
cd /workspace/wadang
export HF_HOME=/workspace/wadang/models/hf HF_HUB_OFFLINE=1
M=work/upload/manifest.csv
O=work/out
mkdir -p $O logs
date -u +%FT%TZ > $O/started.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $O/gpu.txt
venv/bin/pip freeze > $O/pip_freeze.txt
echo "== embed (PE-Core-G)";  venv/bin/python embed.py $M $O/pe_core_emb.npz > logs/embed.log 2>&1; tail -1 logs/embed.log
echo "== segment (Falcon)";   venv/bin/python segment.py $M $O > logs/segment.log 2>&1; tail -1 logs/segment.log
echo "== geometry (MoGe-3)";  venv/bin/python geometry.py $O $M $O > logs/geometry.log 2>&1; tail -1 logs/geometry.log
echo "== synthetic";          venv/bin/python geometry.py $O $M $O --synthetic 30 > logs/synthetic.log 2>&1; tail -1 logs/synthetic.log
date -u +%FT%TZ > $O/finished.txt
echo BATCH_DONE
