# r8 configuration. PROJ is the Slurm account.
export PROJ=${PROJ:-CHANGE_ME}
export R8=${R8:-$SCRATCH/r8}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export ENV_MAIN=${ENV_MAIN:-$SCRATCH/conda-env/r8}        # Julia-1 + Qwen (torch, transformers 5.0.0)
export ENV_LAYA=${ENV_LAYA:-$SCRATCH/conda-env/r8-laya}   # laya 0.3.21
export JULIA_CKPT=${JULIA_CKPT:-$R8/models/Julia-1}
export TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
