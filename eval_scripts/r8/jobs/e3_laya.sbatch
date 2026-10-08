#!/bin/bash
#SBATCH -J r8-e3-laya
#SBATCH -p <gpu-partition>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -o logs/%x-%j.out
# E3b: Laya multilingual and English on the complete samples. Resumable.
set -uo pipefail
source jobs/_env.sh laya
python -u ../r8/e3_laya_full.py --models multilingual,english --tag full
echo "end $(date -Is)"
