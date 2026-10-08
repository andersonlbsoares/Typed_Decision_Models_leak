#!/bin/bash
# Run on a node WITH internet access (e.g. the cluster login node); compute nodes are assumed offline.
# Steps are idempotent:
#   bash stage_login.sh env      # two conda envs (torch cu126 + transformers 5.0.0; laya 0.3.21)
#   bash stage_login.sh models   # Julia-1 and Qwen2.5-7B at the pinned revisions
#   bash stage_login.sh data     # BTZSC, typed-decisions and the 10 datasets -> data/raw/*.parquet
#   bash stage_login.sh samples  # rebuild and VERIFY the audit samples from the stored item ids
#   bash stage_login.sh laya     # Laya weights (pin refs/main of the HF cache to rev. 55cf4c4e afterwards)
set -euo pipefail
cd "$(dirname "$0")"; source ./config.sh; export R8="$PWD"
mkdir -p "$HF_HOME" data logs results
module load <conda-module>
step=${1:?usage: stage_login.sh env|models|data|samples|laya}
case $step in
env)
  [[ -d $ENV_MAIN ]] || conda create -y --prefix "$ENV_MAIN" python=3.12
  source activate "$ENV_MAIN"; export PYTHONNOUSERSITE=1
  pip install --index-url "$TORCH_INDEX" torch
  pip install "transformers==5.0.0" datasets pyarrow huggingface_hub accelerate safetensors \
      sentencepiece protobuf tiktoken numpy scipy pandas scikit-learn
  pip freeze > logs/env_main_freeze.txt; conda deactivate
  [[ -d $ENV_LAYA ]] || conda create -y --prefix "$ENV_LAYA" python=3.12
  source activate "$ENV_LAYA"
  pip install --index-url "$TORCH_INDEX" torch
  pip install "laya==0.3.21" numpy
  pip freeze > logs/env_laya_freeze.txt ;;
models)  source activate "$ENV_MAIN"; export PYTHONNOUSERSITE=1; python -u r8/stage_download.py models ;;
data)    source activate "$ENV_MAIN"; export PYTHONNOUSERSITE=1; python -u r8/stage_download.py data ;;
samples) source activate "$ENV_MAIN"; export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
         (cd code && python -u ../r8/build_samples.py) ;;
laya)
  source activate "$ENV_LAYA"; export USE_TF=0
  python - <<'PY'
import laya, json, os, glob
r = laya.Router(device="cpu", max_loaded=1)
q = {"label": {"type": "choice", "instructions": "Which single label best describes the input text?",
               "criteria": {"c0": "positive", "c1": "negative"}}}
for m in ["english", "multilingual"]:
    print(m, r.predict({"text": "great movie"}, q, model=m)["answers"]["label"]["choice"])
snaps = sorted(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--*", "snapshots", "*")))
json.dump([s for s in snaps if "aya" in s.lower() or "convai" in s.lower()], open("data/laya_snapshots.json", "w"), indent=1)
print("laya snapshots:", open("data/laya_snapshots.json").read())
PY
  ;;
*) echo "unknown step: $step"; exit 1 ;;
esac
echo "[stage_login] $step done"
