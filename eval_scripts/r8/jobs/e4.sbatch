#!/bin/bash
#SBATCH -J r8-e4
#SBATCH -p <gpu-partition>
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=96G
#SBATCH -o logs/%x-%j.out
# E4: deterministic Julia-1. GPU (bs64, bs1, fp32 bs1) and CPU fp32 bs1 run in parallel, each writing its own
# file (r8_e4_gpu / r8_e4_cpu). Resumable.
set -uo pipefail
source jobs/_env.sh main
python -u ../r8/e4_julia_det.py --modes cpu_fp32_bs1 --threads 12 --tag cpu > ../logs/e4_cpu_$SLURM_JOB_ID.log 2>&1 &
CPU=$!
python -u ../r8/e4_julia_det.py --modes gpu_bs64,gpu_bs1,gpu_fp32_bs1 --tag gpu
wait $CPU; echo "cpu exit $?"; tail -n 8 ../logs/e4_cpu_$SLURM_JOB_ID.log
echo "end $(date -Is)"
