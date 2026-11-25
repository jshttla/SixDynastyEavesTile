#!/bin/bash
# v2 cloud environment: Falcon-Perception (segmentation) + MoGe-2 (geometry). Grounding DINO / SAM 2.1 removed.
set -x
source /etc/network_turbo 2>/dev/null || true
W=/workspace/wadang
cd $W
/root/miniconda3/bin/python -m venv $W/venv
. $W/venv/bin/activate
pip install -q -U pip wheel setuptools
pip install -q torch torchvision
pip install -q "transformers>=4.57" accelerate huggingface_hub safetensors einops pycocotools opencv-python-headless pillow numpy scipy timm
pip install -q git+https://github.com/microsoft/MoGe.git || echo MOGE_FAIL
python -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
export HF_HOME=$W/models/hf HF_HUB_DISABLE_XET=1
hf download tiiuae/Falcon-Perception --revision 19-08-2026 2>&1 | tail -1
hf download Ruicheng/moge-3-vitg
hf download timm/vit_pe_core_gigantic_patch14_448.fb 2>&1 | tail -1
pip freeze > $W/logs/pip_freeze_v2.txt
echo SETUP_V2_DONE
