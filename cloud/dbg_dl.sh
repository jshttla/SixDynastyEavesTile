cd /workspace/wadang
source /etc/network_turbo >/dev/null 2>&1
export HF_HOME=/workspace/wadang/models/hf HF_HUB_DISABLE_XET=1
timeout 100 venv/bin/hf download facebook/dinov2-large config.json 2>&1 | tail -3
echo "--- direct HF (turbo)"
timeout 60 curl -sIL https://huggingface.co/facebook/dinov2-large/resolve/main/model.safetensors 2>&1 | grep -iE "^HTTP|^location|content-length" | head -8
echo "--- mirror (no proxy)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
timeout 60 curl -sIL https://hf-mirror.com/facebook/dinov2-large/resolve/main/model.safetensors 2>&1 | grep -iE "^HTTP|^location|content-length" | head -8
