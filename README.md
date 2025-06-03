# Bias Detection and Mitigation in Large Language Models

## Running evaluation

Using the evaluation suite of 279 classification tasks taken from Super-NaturalInstructions (Wang et al., 2022), use the following command to install the necessary packages before doing the label bias assessment for Huggingface models:

```bash
pip install -r requirements.txt
```

Then use the following script to download and prepare the evaluation data:

```bash
./scripts/prepare_superni_data.sh
```

To run evaluation, see the scrips under `./scripts`. For example, you can use the following command to run evaluation for DeepSeek-R1-Distill-Llama-8B:

```csh
python -m src.superni.run_completions_eval \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --data_dir data/eval/superni/splits/classification_tasks/ --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples 8 \
    --eval_bias_score --eval_looc --eval_cc --eval_dc \
    --max_num_instances_per_eval_task 100 --output_dir runs/DeepSeek-R1-Distill-Llama-8B/8_shots/
```
