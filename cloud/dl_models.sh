cd /workspace/wadang
source /etc/network_turbo >/dev/null 2>&1
export HF_HOME=/workspace/wadang/models/hf HF_HUB_DISABLE_XET=1
for m in facebook/dinov2-large Ruicheng/moge-2-vitl-normal facebook/sam2.1-hiera-large IDEA-Research/grounding-dino-base; do
  echo "=== $m $(date -u +%T)"
  venv/bin/hf download $m --include "*.json" "*.safetensors" "*.txt" "*.pt" "*.bin" "*.model" 2>&1 | tail -2
done
du -sh models/hf/hub/*
echo DL_DONE
