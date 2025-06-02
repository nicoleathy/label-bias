#!/bin/csh

setenv HF_TOKEN ""

# Set model configuration
set model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
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


