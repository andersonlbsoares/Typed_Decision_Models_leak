#!/bin/bash
#SBATCH -J r8-smoke
#SBATCH -p <gpu-dev-partition>
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=96G
#SBATCH -o logs/%x-%j.out
set -uo pipefail
source jobs/_env.sh main
python -u ../r8/smoke.py
source activate "$ENV_LAYA"
python -u ../r8/e3_laya_full.py --models multilingual --keys emotion --limit 100 --tag smoke 2>&1 | tee ../results/smoke_laya.log
source activate "$ENV_MAIN"; python ../r8/project.py | tee ../results/smoke_projection.txt
echo "end $(date -Is)"
