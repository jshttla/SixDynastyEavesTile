cd /workspace/wadang
source /etc/network_turbo >/dev/null 2>&1
export HF_HOME=/workspace/wadang/models/hf HF_HUB_DISABLE_XET=1
venv/bin/hf download facebook/dinov2-large config.json preprocessor_config.json 2>&1 | tail -1
venv/bin/hf download IDEA-Research/grounding-dino-base 2>&1 | tail -1
venv/bin/hf download Ruicheng/moge-2-vitl-normal 2>&1 | tail -1
for d in models/hf/hub/models--*/snapshots/*/; do echo "$d: $(ls $d | tr '\n' ' ')"; done
