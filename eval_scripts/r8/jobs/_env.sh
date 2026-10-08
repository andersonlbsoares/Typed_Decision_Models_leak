# Sourced by every job: offline mode, env activation and context print. Arg 1: main|laya
cd "${R8_ROOT:-$SLURM_SUBMIT_DIR}"; source ./config.sh; export R8="$PWD"
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_JOB_NODELIST  start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
module load <conda-module>
if [[ ${1:-main} == laya ]]; then source activate "$ENV_LAYA"; else source activate "$ENV_MAIN"; fi
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
mkdir -p results logs; cd code      # run_julia creates ./out here
