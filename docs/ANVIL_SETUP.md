# Purdue Anvil environment setup

This guide records the environment used for the CSC 8851 adaptive-pruning project. The verified baseline ran on one NVIDIA A100-SXM4-40GB GPU.

## Repository and storage layout

Clone the repository with its submodules into project storage. Replace the example allocation and user paths with your own Anvil locations.

```bash
export ALLOCATION=cis260919
export PROJECT_ROOT=/anvil/projects/x-$ALLOCATION/cdonthireddy/3dgs
export SCRATCH_ROOT=/anvil/scratch/x-cdonthireddy/3dgs

mkdir -p "$PROJECT_ROOT" "$SCRATCH_ROOT/datasets"
cd "$PROJECT_ROOT"
git clone --recursive git@github.com:Chandra-Sai-Reddy/adaptive-importance-pruning-3dgs.git gaussian-splatting
cd gaussian-splatting
```

Datasets and trained checkpoints are intentionally excluded from Git because they are large. The Bicycle scene used here occupies about 2.6 GB, and the final 30,000-iteration model directory occupies about 12 GB.

## Modules

The successful Anvil environment used:

```bash
module --force purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0
module load conda/2026.03
source /apps/anvil/external/apps/conda/2026.03/etc/profile.d/conda.sh
```

Verified software versions were Python 3.10.22, PyTorch 2.8.0+cu128, and CUDA runtime 12.8. GCC 9 or newer is required to compile the PyTorch extensions; GCC 11.2.0 was used.

## Conda environment

```bash
export ENV_PATH="$PROJECT_ROOT/envs/gaussian-splatting"
conda create -p "$ENV_PATH" python=3.10 -y
conda activate "$ENV_PATH"

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 plyfile tqdm tensorboard joblib opencv-python-headless
```

## Compile CUDA extensions

```bash
cd "$PROJECT_ROOT/gaussian-splatting"
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/fused-ssim
```

The A100 has compute capability 8.0. Change `TORCH_CUDA_ARCH_LIST` when compiling for a different GPU architecture.

PyTorch extension imports may require its shared-library directory at runtime:

```bash
export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

## Validate on a compute node

Do not test CUDA availability on a login node. Submit the included smoke test:

```bash
mkdir -p "$PROJECT_ROOT/logs"
sbatch slurm/gpu_smoke.slurm
```

A successful run prints the detected GPU and ends with `ALL GPU SMOKE TESTS PASSED`. The scripts currently contain the paths and allocation used for the reported experiment; adapt the `#SBATCH -A`, log paths, environment path, repository path, dataset path, and output path for another Anvil account.
