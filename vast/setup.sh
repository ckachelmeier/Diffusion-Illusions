#!/usr/bin/env bash
# One-time setup of a vast.ai instance (or any Linux machine with an NVIDIA GPU) for this project.
#     bash vast/setup.sh
# Safe to run again: anything already done is skipped or quick. See vast/README.md for the whole workflow.
#
# Optional environment variables:
#     TORCH_CUDA=cu126|cu130    CUDA build of torch to install. Default: cu130 for Blackwell GPUs (RTX 50xx, B200), else cu126
#     SKIP_MODEL_DOWNLOAD=1     Don't download the Stable Diffusion weights now; the first run downloads them instead
#     HF_TOKEN=...              Hugging Face token, only needed for a model that requires one
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

#Same versions as the Windows .venv this project was developed with
PYTHON_VERSION=3.10   #The pins in requirements.txt have no wheels for 3.12+
TORCH_VERSION=2.14.1
TORCHVISION_VERSION=0.29.1
VENV_PY="$PROJECT_DIR/.venv/bin/python"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. This needs an instance with an NVIDIA GPU."

step "System packages"
#opencv-python needs libGL and glib, which slim Docker images leave out. tmux keeps runs going when SSH disconnects.
missing=()
ldconfig -p | grep -q 'libGL\.so\.1' || missing+=(libgl1)
ldconfig -p | grep -q 'libgthread-2\.0\.so\.0' || missing+=(libglib2.0-0)
command -v tmux >/dev/null || missing+=(tmux)
if ((${#missing[@]})); then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO=sudo
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
else
    echo "Already installed"
fi

step "uv (Python package manager)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "Python $PYTHON_VERSION environment in .venv"
if [ -d .venv ] && [ ! -x "$VENV_PY" ]; then
    fail ".venv exists but isn't a Linux environment (copied over from Windows?). Delete it and run this again."
fi
[ -x "$VENV_PY" ] || uv venv --python "$PYTHON_VERSION" .venv
"$VENV_PY" --version

step "PyTorch"
read -r gpu_name compute_cap driver < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 | awk -F', ' '{gsub(/ /, "_", $1); print $1, $2, $3}')
echo "GPU: ${gpu_name//_/ } (compute capability $compute_cap), driver $driver"
if [ -z "${TORCH_CUDA:-}" ]; then
    #torch's cu126 builds stop at Hopper (compute capability 9.x); Blackwell needs a newer CUDA build
    if [ "${compute_cap%%.*}" -ge 10 ]; then TORCH_CUDA=cu130; else TORCH_CUDA=cu126; fi
fi
if [ "$TORCH_CUDA" = cu130 ] && [ "${driver%%.*}" -lt 580 ]; then
    echo "warning: CUDA 13 builds of torch need NVIDIA driver 580 or newer, and this machine has $driver. Pick another instance if the check at the end fails."
fi
uv pip install --python "$VENV_PY" \
    "torch==$TORCH_VERSION+$TORCH_CUDA" "torchvision==$TORCHVISION_VERSION+$TORCH_CUDA" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

step "Project requirements"
#constraints.txt pins everything else to the versions in the Windows .venv, so runs behave the same here
uv pip install --python "$VENV_PY" -r requirements-local.txt -c vast/constraints.txt

if [ "${SKIP_MODEL_DOWNLOAD:-0}" != 1 ]; then
    step "Stable Diffusion weights (a few GB, cached for later runs)"
    #Loads the model the same way source/stable_diffusion.py does, so exactly the files it needs get downloaded
    "$VENV_PY" - <<'EOF'
import dataclasses

import torch
from diffusers import StableDiffusionPipeline

from illusion_gen.config import RunConfig

model = {field.name: field.default for field in dataclasses.fields(RunConfig)}['model']
print('Model:', model, flush=True)
StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float, requires_safety_checker=False, safety_checker=None)
EOF
fi

step "Checking the environment"
"$VENV_PY" - <<'EOF'
import torch

assert torch.cuda.is_available(), 'torch cannot see the GPU'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device='cuda')
(x @ x).sum().item()  #Raises "no kernel image is available" if this torch build doesn't support the GPU

import illusion_gen.runner
import source.stable_diffusion
print('Project imports OK')
EOF

cat <<EOF

Setup finished. Start runs with:
    cd $PROJECT_DIR
    bash vast/run.sh --config runs.yaml --dry-run    (check the config)
    bash vast/run.sh --config runs.yaml
EOF
