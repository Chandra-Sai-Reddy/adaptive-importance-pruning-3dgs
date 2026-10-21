#!/bin/bash
#SBATCH -A cis260919-gpu
#SBATCH -p gpu
#SBATCH -J bicycle-baseline
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.out
#SBATCH -e /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.err

set -euo pipefail

module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0
module load conda/2026.03

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH="/anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

REPO=/anvil/projects/x-cis260919/cdonthireddy/3dgs/gaussian-splatting
DATA=/anvil/scratch/x-cdonthireddy/3dgs/datasets/mipnerf360/bicycle
OUTPUT=/anvil/projects/x-cis260919/cdonthireddy/3dgs/outputs/bicycle/baseline

mkdir -p "$OUTPUT"
cd "$REPO"

echo "Job started: $(date)"
nvidia-smi

python -u train.py \
    -s "$DATA" \
    -m "$OUTPUT" \
    --eval \
    --data_device cpu \
    --iterations 30000 \
    --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000

echo "Job completed: $(date)"
du -sh "$OUTPUT"
