#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=120         # Number of CPU to request for the job
#SBATCH --mem=400GB                 # How much memory does your job require?
#SBATCH --gres=gpu:4                # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00   
                                    # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out         # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory


################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchlong          # The partition you've been assigned
#SBATCH --account=tanahhweeresearch            # The account you've been assigned (normally student)
#SBATCH --qos=nicolet.2023-2024-11-21    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=                           # Who should receive the email notifications
#SBATCH --job-name=label-bias                     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/11.8.0

PY=$(which python3.11)
echo "using: $PY"
$PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

echo "Logging into Hugging Face..."
hf auth login --token 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
echo "Login completed successfully!"

# If you require any packages, install it as usual before the srun job submission.
# pip install matplotlib
# pip install seaborn
# pip install scikit-metrics
# pip install -r requirements.txt
# pip install flash_attn==2.5.8
# pip install torch==2.3.1
# pip install accelerate>=0.26.0
# pip install transformers==4.46.0
# pip install datasets

# MODELS=(
# "meta-llama/Llama-3.2-3B"
# "meta-llama/Llama-3.2-3B-Instruct"
# "microsoft/Phi-3.5-mini-instruct"
# "microsoft/Phi-3.5-MoE-instruct"
# "mistralai/Mixtral-8x7B-v0.1"
# "mistralai/Mixtral-8x7B-Instruct-v0.1"
# "Qwen/Qwen2.5-32B"
# "Qwen/Qwen2.5-32B-Instruct"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# "google/gemma-3-4b-pt"
# "google/gemma-3-4b-it"
# )

# # Submit your job to the cluster
# srun --gres=gpu:4 python -m src.superni.run_completions_eval \
#     --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --data_dir data/eval/superni/splits/classification_tasks/ --task_dir data/eval/superni/classification_tasks/ \
#     --num_pos_examples 8 \
#     --eval_bias_score --eval_looc --eval_cc --eval_dc \
#     --max_num_instances_per_eval_task 100 --output_dir runs/Mixtral-8x7B-Instruct-v0.1/8_shots/

srun --gres=gpu:4 $PY -m src.superni.run_completions_eval \
    --model Qwen/Qwen2.5-32B-Instruct \
    --data_dir data/eval/superni/splits/classification_tasks/ \
    --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples 4 \
    --eval_bias_score --eval_looc --eval_cc --eval_dc \
    --max_num_instances_per_eval_task 100 \
    --output_dir runs_rerun/Qwen2.5-32B-Instruct/4_shots/