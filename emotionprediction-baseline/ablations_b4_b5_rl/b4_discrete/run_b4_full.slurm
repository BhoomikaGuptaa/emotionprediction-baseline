#!/bin/bash
#SBATCH --job-name=b4-full
#SBATCH --partition=nsfqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/b4-full-%j.log
#SBATCH --error=logs/b4-full-%j.log

set -euo pipefail

cd "$HOME/prediction/b4_final/b4_discrete"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/prediction/python_packages/b4-grpo/transformers_shim:$HOME/prediction/python_packages/b4-grpo"

CONTAINER="$HOME/prediction/verl_env_pinned.sif"
MODEL_DIR="$HOME/prediction/models/Qwen2.5-3B-Instruct"
DATA_PATH="$HOME/prediction/b4_final/IEMOCAP_features.pkl"
OUTPUT_DIR="$HOME/prediction/b4_final/b4_discrete/outputs/b4_full_seed42"

echo "===== B4 FULL RUN ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

echo "===== INPUT CHECKS ====="
test -f "$DATA_PATH"
test -f "$MODEL_DIR/config.json"
test -f "$CONTAINER"
test -d "$HOME/prediction/python_packages/b4-grpo/trl"

du -sh "$MODEL_DIR"
ls -lh "$DATA_PATH"

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists:"
    echo "$OUTPUT_DIR"
    echo "Refusing to overwrite an earlier run."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "===== PACKAGE VERSIONS ====="
apptainer exec --nv "$CONTAINER" \
env PYTHONPATH="$PYTHONPATH" \
python - <<'PY'
import torch
import transformers
import trl
import peft
import datasets
import accelerate
import numpy

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("peft:", peft.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("numpy:", numpy.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated job.")

print("gpu_name:", torch.cuda.get_device_name(0))

from trl import GRPOConfig, GRPOTrainer
print("GRPO imports: OK")
PY

echo "===== LOCAL INVARIANT TEST ====="
apptainer exec --nv "$CONTAINER" \
env PYTHONPATH="$PYTHONPATH" \
python run.py --mode self_test

echo "===== TRAINING START ====="
date

apptainer exec --nv "$CONTAINER" \
env PYTHONPATH="$PYTHONPATH" \
python run.py \
  --mode train \
  --data_path "$DATA_PATH" \
  --base_model "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42 \
  --sft_epochs 2.0 \
  --grpo_steps 300 \
  --grpo_g 2 \
  --save_steps 100 \
  --batch_size 2 \
  --grad_accum 4 \
  --max_seq_len 1024 \
  --max_new_tokens 96

echo "===== REQUIRED OUTPUT CHECKS ====="

test -f "$OUTPUT_DIR/run_metadata.json"
echo "run_metadata.json: OK"

test -f "$OUTPUT_DIR/dev_checkpoint_scores.json"
echo "dev_checkpoint_scores.json: OK"

test -d "$OUTPUT_DIR/best_dev_model"
echo "best_dev_model: OK"

echo "===== OUTPUT SIZE ====="
du -sh "$OUTPUT_DIR"

echo "===== IMPORTANT FILES ====="
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name '*.json' -o -name 'adapter_model.safetensors' -o -name 'model.safetensors.index.json' \) \
  -printf '%p %s bytes\n' | sort

echo "===== B4 FULL RUN COMPLETED ====="
date
