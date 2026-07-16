#!/bin/csh

setenv HF_TOKEN "INSERT_YOUR_TOKEN"
setenv PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"
setenv WANDB_DISABLED "true"
setenv PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

set model_name="meta-llama/Llama-3.2-3B" 
set num_pos_examples=8
set max_instances=300
set max_source_length=2000
set max_target_length=47
set timestamp=`date +'%Y%m%d_%H%M%S'`
set output_dir="runs/${model_name}/${num_pos_examples}_shots/${timestamp}"

# Path to filtered task list
set tasks_file="test_tasks_top20.txt"

# Run evaluation only on the top-20 tasks
python -m src.superni.run_completions_eval \
   --data_dir data/eval/superni/splits/classification_tasks/ \
   --task_dir data/eval/superni/classification_tasks/ \
   --tasks_file $tasks_file \
   --max_num_instances_per_eval_task $max_instances \
   --max_source_length $max_source_length \
   --max_target_length $max_target_length \
   --num_pos_examples ${num_pos_examples} \
   --eval_bias_score --eval_looc --eval_cc --eval_dc \
   --model $model_name \
   --output_dir "${output_dir}" \
   --logging_dir "logs/${model_name}/${num_pos_examples}_shots/"



