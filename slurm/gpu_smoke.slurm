#!/bin/bash
#SBATCH -A cis260919-gpu
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
#SBATCH -J 3dgs-smoke
#SBATCH -o /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.out
#SBATCH -e /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0
module load conda/2026.03

source /apps/anvil/external/apps/conda/2026.03/etc/profile.d/conda.sh
conda activate /anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH="/anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_LAUNCH_BLOCKING=1

cd /anvil/projects/x-cis260919/cdonthireddy/3dgs/gaussian-splatting

hostname
nvidia-smi

python - <<'PY'
import torch
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
import diff_gaussian_rasterization

assert torch.cuda.is_available(), "PyTorch cannot access the GPU"

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

points = torch.randn((1024, 3), device="cuda", dtype=torch.float32)
distances = distCUDA2(points)
torch.cuda.synchronize()
print("simple-knn:", distances.shape, "OK")

image = torch.rand((1, 3, 64, 64), device="cuda")
score = fused_ssim(image, image)
torch.cuda.synchronize()
print("fused-ssim:", float(score), "OK")

print("rasterizer import: OK")
print("ALL GPU SMOKE TESTS PASSED")
PY
