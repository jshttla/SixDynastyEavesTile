cd /workspace/wadang
export HF_HOME=/workspace/wadang/models/hf HF_HUB_OFFLINE=1
venv/bin/python smoke.py smoke_test.jpg > logs/smoke.log 2>&1
grep -E "^(GDINO|SAM2|DINOV2|MOGE2)" logs/smoke.log
rm -rf work/test_out
venv/bin/python segment.py test_manifest.csv work/test_out > logs/seg_test.log 2>&1
tail -2 logs/seg_test.log
venv/bin/python geometry.py work/test_out test_manifest.csv work/test_out > logs/geom_test.log 2>&1
tail -2 logs/geom_test.log
