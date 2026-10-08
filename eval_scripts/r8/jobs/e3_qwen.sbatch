#!/bin/bash
#SBATCH -J r8-e3-qwen
#SBATCH -p <gpu-partition>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96G
#SBATCH -o logs/%x-%j.out
# E3a: Qwen2.5-7B-Instruct on the complete samples of the 10 datasets (39,066 prompts). Resumable.
set -uo pipefail
source jobs/_env.sh main
python -u ../r8/e3_qwen_full.py --tag full
echo "end $(date -Is)"
