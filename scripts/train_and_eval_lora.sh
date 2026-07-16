# Load Hugging Face token securely
setenv HF_TOKEN "INSERT_YOUR_TOKEN"
setenv PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"
setenv WANDB_DISABLED "true"
setenv PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

# Model and LoRA configuration
set model_name="meta-llama/Llama-3.2-3B" 
set num_pos_examples=8
set seed=42

# Add timestamp to output directory for better experiment tracking
set timestamp=`date +'%Y%m%d_%H%M%S'`
set output_dir="runs/${model_name}/lora_${num_pos_examples}_shots/${timestamp}/"

# Path to filtered task list
set tasks_file="test_tasks_top20.txt"

# Run LoRA-tuned evaluation on top-20 tasks
python -m src.superni.run_lora_completions_eval \
    --data_dir data/eval/superni/splits/classification_tasks/ \
    --task_dir data/eval/superni/classification_tasks/ \
    --tasks_file $tasks_file \
    --max_num_instances_per_eval_task 100 \
    --max_source_length 2000 --max_target_length 47 --max_seq_length 2048 \
    --num_pos_examples ${num_pos_examples} --add_task_definition True \
    --eval_bias_score --eval_cc --eval_dc --eval_looc \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 \
    --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 \
    --num_train_epochs 5 --learning_rate 0.0002 --warmup_ratio 0.06 \
    --max_grad_norm 0.3 --weight_decay 0.03 --adam_beta2 0.999 \
    --bf16 --gradient_checkpointing \
    --logging_steps 10 --logging_dir "logs/${model_name}/lora_${num_pos_examples}_shots/" \
    --overwrite_output_dir --ddp_find_unused_parameters=False \
    --seed $seed \
    --model $model_name --output_dir "${output_dir}"





