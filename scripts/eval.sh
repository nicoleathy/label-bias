#!/bin/csh

setenv HF_TOKEN ""

# set model_name="meta-llama/Llama-3.2-3B"
# set model_name="meta-llama/Llama-3.2-3B-Instruct"
# set model_name="google/gemma-2-27b"
# set model_name="mistralai/Mixtral-8x7B-v0.1"
# set model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
# set model_name="Qwen/Qwen2.5-72B" 
# set model_name="mistralai/Mistral-7B-v0.1"
# set model_name="mistralai/Mistral-7B-Instruct-v0.1"
# set model_name="meta-llama/Llama-2-7b-hf"
# set model_name="meta-llama/Llama-2-7b-chat-hf"

# set num_pos_examples=8
# set output_dir="runs/${model_name}/${num_pos_examples}_shots/"

# python -m src.superni.run_completions_eval \
#    --data_dir data/eval/superni/splits/classification_tasks/ --task_dir data/eval/superni/classification_tasks/ \
#    --max_num_instances_per_eval_task 100 --max_source_length 2000 --max_target_length 47 \
#    --num_pos_examples ${num_pos_examples} \
#    --eval_bias_score --eval_looc --eval_cc --eval_dc \
#    --model $model_name --output_dir "${output_dir}"

# Set model configuration
set model_name="Qwen/Qwen2.5-32B-Instruct" 
set num_pos_examples=8
set max_instances=300
set max_source_length=2000
set max_target_length=47
set timestamp=$(date +'%Y%m%d_%H%M%S')
set output_dir="runs/${model_name}/${num_pos_examples}_shots/${timestamp}"

# Run evaluation
python -m src.superni.run_completions_eval \
   --data_dir data/eval/superni/splits/classification_tasks/ \
   --task_dir data/eval/superni/classification_tasks/ \
   --max_num_instances_per_eval_task $max_instances \
   --max_source_length $max_source_length \
   --max_target_length $max_target_length \
   --num_pos_examples ${num_pos_examples} \
   --eval_bias_score --eval_looc --eval_cc --eval_dc \
   --model $model_name \
   --output_dir "${output_dir}" \
   --logging_dir "logs/${model_name}/${num_pos_examples}_shots/" 


