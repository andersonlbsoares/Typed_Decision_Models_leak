#!/bin/bash
#SBATCH -J r8-e1
#SBATCH -p <gpu-partition>
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --array=0-3
#SBATCH -o logs/%x-%A_%a.out
# E1: probe A under diluted exposure. One array task per dataset (1 GPU each); 6 rho x 5 seeds = 30 fine-tunings
# per dataset. Resumable (completed runs are skipped).
set -uo pipefail
KEYS=(emotion agnews yelp yahoo)
source jobs/_env.sh main
python -u ../r8/e1_julia_dilution.py "${KEYS[$SLURM_ARRAY_TASK_ID]}" --seeds 0,1,2,3,4 --rhos 0,0.05,0.1,0.25,0.5,1
echo "end $(date -Is)"
