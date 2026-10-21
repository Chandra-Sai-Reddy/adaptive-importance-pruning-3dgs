#!/bin/bash
#SBATCH -A cis260919-gpu
#SBATCH -p gpu
#SBATCH -J bicycle-eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.out
#SBATCH -e /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs/%x-%j.err

set -euo pipefail

module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0
module load conda/2026.03

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting

export LD_LIBRARY_PATH="/anvil/projects/x-cis260919/cdonthireddy/envs/gaussian-splatting/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

REPO=/anvil/projects/x-cis260919/cdonthireddy/3dgs/gaussian-splatting
MODEL=/anvil/projects/x-cis260919/cdonthireddy/3dgs/outputs/bicycle/baseline

cd "$REPO"

echo "Evaluation started: $(date)"
nvidia-smi

python -u render.py \
    -m "$MODEL" \
    --iteration 30000 \
    --skip_train

python -u metrics.py -m "$MODEL"

echo "Evaluation completed: $(date)"
du -sh "$MODEL"
