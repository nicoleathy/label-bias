# Label Bias Calibration in LLMs Depends on Label-Space Size

Code for the paper **"Label Bias Calibration in LLMs Depends on Label-Space Size"**.

We compare Contextual Calibration (CC), Domain Contextual Calibration (DCC), and Leave-One-Out Calibration (LOOC) across 279 Super-NaturalInstructions classification tasks and 10 LLMs from 5 families. LOOC performs best overall, but CC or DCC is better when the label space is large (K ≥ 6 with k = 8 demonstrations), because LOOC only sees labels that appear among its demonstrations.

**Selection rule:** use LOOC when the demonstrations reliably cover the label space, and CC or DCC otherwise.

## Setup

```bash
pip install -r requirements.txt
./scripts/prepare_superni_data.sh
```

## Running evaluation

Scripts for each model are in `./scripts`. Example:

```bash
python -m src.superni.run_completions_eval \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --data_dir data/eval/superni/splits/classification_tasks/ \
    --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples 8 \
    --eval_bias_score --eval_looc --eval_cc --eval_dc \
    --max_num_instances_per_eval_task 100 \
    --output_dir runs/DeepSeek-R1-Distill-Llama-8B/8_shots/
```

`--num_pos_examples` sets the number of demonstrations (k). Set it to 4 to reproduce the demonstration budget experiment.

## Citation

```bibtex
@inproceedings{anonymous2027labelbias,
  title     = {Label Bias Calibration in LLMs Depends on Label-Space Size},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```
