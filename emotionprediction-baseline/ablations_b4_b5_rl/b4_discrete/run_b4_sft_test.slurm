#!/bin/bash
#SBATCH --job-name=b4-sft-test
#SBATCH --partition=nsfqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/b4-sft-test-%j.log
#SBATCH --error=logs/b4-sft-test-%j.log

set -euo pipefail

cd "$HOME/prediction/b4_final/b4_discrete"

export PYTHONPATH="$HOME/prediction/python_packages/b4-grpo/transformers_shim:$HOME/prediction/python_packages/b4-grpo"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CONTAINER="$HOME/prediction/verl_env_pinned.sif"
DATA_PATH="$HOME/prediction/b4_final/IEMOCAP_features.pkl"
MODEL_PATH="$HOME/prediction/b4_final/b4_discrete/outputs/b4_full_seed42/sft_merged"
SAVE_PATH="$HOME/prediction/b4_final/b4_discrete/outputs/b4_full_seed42/sft_only_predictions.json"

nvidia-smi

apptainer exec --nv "$CONTAINER" \
env PYTHONPATH="$PYTHONPATH" \
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable")
PY

apptainer exec --nv "$CONTAINER" \
env PYTHONPATH="$PYTHONPATH" \
python run.py \
  --mode eval \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --split test \
  --save_path "$SAVE_PATH"

test -f "$SAVE_PATH"
echo "Predictions saved:"
ls -lh "$SAVE_PATH"

echo "===== B4 TEST EVALUATION COMPLETED ====="
