#!/usr/bin/env bash
# Run uncalibrated scoring, CC, DCC, LOOC and BiasScore for one or more models.
# Run from the repository root after scripts/prepare_superni_data.sh.
#
# Usage:
#   export HF_TOKEN=...            # only needed for gated models (e.g. Llama)
#   bash scripts/run_eval.sh                       # all 10 models, k = 8
#   K=4 bash scripts/run_eval.sh Qwen/Qwen2.5-32B-Instruct   # k = 4 ablation (Table 3)
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

K="${K:-8}"                 # number of demonstrations per prompt
MAX_INSTANCES="${MAX_INSTANCES:-100}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/Phi-3.5-MoE-instruct"
    "mistralai/Mixtral-8x7B-v0.1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "Qwen/Qwen2.5-32B"
    "Qwen/Qwen2.5-32B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  )
fi

for MODEL in "${MODELS[@]}"; do
  NAME="${MODEL##*/}"
  python -m src.superni.run_completions_eval \
    --model "$MODEL" \
    --data_dir data/eval/superni/splits/classification_tasks/ \
    --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples "$K" \
    --max_num_instances_per_eval_task "$MAX_INSTANCES" \
    --eval_bias_score --eval_cc --eval_dc --eval_looc \
    --output_dir "runs/${NAME}/${K}_shots/"
done
