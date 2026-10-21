# Reproducing the Bicycle baseline

This document records the completed baseline experiment used in the preliminary report. It does not claim adaptive-pruning results: the pruning implementation and controlled comparisons remain future work.

## Inputs

- Code: this repository, including all Git submodules
- Dataset: Mip-NeRF 360 Bicycle scene
- Dataset project page: https://jonbarron.info/mipnerf360/
- Verified scene directory: `/anvil/scratch/x-cdonthireddy/3dgs/datasets/mipnerf360/bicycle`

The scene directory must contain `images/`, `images_2/`, `images_4/`, `images_8/`, and `sparse/0/`. The extracted Bicycle scene occupied approximately 2.6 GB.

## Before submitting jobs

Follow [ANVIL_SETUP.md](ANVIL_SETUP.md), then edit the account, log, environment, repository, dataset, and model paths near the top of each file in `slurm/`. Create the log directory before submission.

```bash
mkdir -p /anvil/projects/x-cis260919/cdonthireddy/3dgs/logs
```

## 1. GPU smoke test

```bash
sbatch slurm/gpu_smoke.slurm
```

The test validates CUDA access plus the `simple_knn`, `fused_ssim`, and `diff_gaussian_rasterization` extensions. The verified run used an NVIDIA A100-SXM4-40GB and ended with `ALL GPU SMOKE TESTS PASSED`.

## 2. Train the baseline

```bash
TRAIN_JOB=$(sbatch --parsable slurm/train_bicycle_baseline.slurm)
echo "$TRAIN_JOB"
```

The training script uses one GPU, 16 CPU cores, 64 GB host memory, and a six-hour limit. It trains for 30,000 iterations with the standard LLFF holdout split, keeps image data on the CPU, evaluates at iterations 7,000, 15,000, and 30,000, and saves checkpoints and Gaussian point clouds at the same iterations.

The successful reported training job was `21565838`. It completed in 36 minutes 42 seconds with exit code `0:0`.

## 3. Render and evaluate

Submit evaluation after training succeeds:

```bash
sbatch --dependency=afterok:$TRAIN_JOB slurm/evaluate_bicycle_baseline.slurm
```

For a previously completed model, submit the evaluation script directly:

```bash
sbatch slurm/evaluate_bicycle_baseline.slurm
```

The script renders the held-out test views at iteration 30,000 and runs `metrics.py`. The successful reported evaluation job was `21566075`; it completed in 3 minutes 3 seconds with exit code `0:0`.

## Verified baseline results

| Quantity | Value |
|---|---:|
| PSNR | 25.1361 dB |
| SSIM | 0.7473 |
| LPIPS | 0.2422 |
| Final Gaussian primitives | 4,770,285 |
| Final `point_cloud.ply` size | 1.2 GB |
| Complete model-directory size | 12 GB |
| Training time | 36 min 42 s |
| Evaluation time | 3 min 3 s |

The final point cloud is located under `point_cloud/iteration_30000/point_cloud.ply` inside the configured model directory. Test renderings and metric inputs are written beneath that model directory by `render.py`.

## Monitor and inspect jobs

```bash
squeue -j "$TRAIN_JOB" -o "%.18i %.18j %.12P %.10T %.10M %.40R"
sacct -j "$TRAIN_JOB" --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode
```

Job output and error logs use the `%x-%j.out` and `%x-%j.err` pattern in the configured log directory.

## Reproducibility boundary

The repository intentionally excludes the 2.6 GB dataset, the 12 GB trained model directory, downloaded VGG/LPIPS weights, and scheduler logs. The exact job scripts, software configuration, commands, job identifiers, timings, and aggregate metrics are retained here. Numerical results can vary slightly with GPU model, dependency revisions, and nondeterministic CUDA operations.
