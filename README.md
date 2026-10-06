# Label Bias Calibration in LLMs Depends on Label-Space Size

This repository contains the code for the paper **"Label Bias Calibration in LLMs Depends on Label-Space Size"**.

We compare three label bias calibration methods, Contextual Calibration (CC), Domain Contextual Calibration (DCC), and Leave-One-Out Calibration (LOOC), across 279 Super-NaturalInstructions classification tasks and 10 LLMs from 5 model families, using one common evaluation pipeline.

## Key findings

- **Overall**, LOOC performs best for every model on Macro-F1, RSD, and BiasScore.
- **By label-space size K**, the ranking changes. On Macro-F1 and BiasScore, LOOC is best on all ten models when K = 2, on eight of ten when 3 ≤ K ≤ 5, and on none when K ≥ 6, where CC or DCC is consistently better.
- **Label coverage explains the reversal.** LOOC estimates label preferences from the k demonstrations, so it only sees labels that appear among them. CC and DCC estimate preferences over the full label set.

| Label-space size | Tasks | LOOC margin (Macro-F1) | Prompts covering all labels (k = 8) |
|---|---|---|---|
| K = 2 | 197 | +4.18 | 98% |
| 3 ≤ K ≤ 5 | 60 | +1.24 | 58% |
| K ≥ 6 | 22 | -3.34 | 8% or less |

LOOC margin is the average Macro-F1 difference between LOOC and the better of CC and DCC.

### Selection rule

Use **LOOC** when the demonstrations reliably cover the label space (K ≤ 5 with k = 8). Use **CC or DCC** otherwise. For larger label spaces, CC and DCC are both cheaper (about k times less compute) and more accurate.

## Setup

We used Python 3.11, PyTorch 2.4, and Transformers 4.46. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

We use the evaluation suite of 279 classification tasks from Super-NaturalInstructions (Wang et al., 2022), the same suite used by Reif and Schwartz (2024). Download and prepare the data with:

```bash
./scripts/prepare_superni_data.sh
```

## Running evaluation

Example scripts for each model are in `./scripts`. For example, to evaluate DeepSeek-R1-Distill-Llama-8B with k = 8 demonstrations:

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

### Main arguments

| Argument | Description |
|---|---|
| `--model` | Hugging Face model ID |
| `--num_pos_examples` | Number of demonstrations per prompt (k) |
| `--eval_bias_score` | Compute label bias metrics |
| `--eval_cc` | Run Contextual Calibration |
| `--eval_dc` | Run Domain Contextual Calibration |
| `--eval_looc` | Run Leave-One-Out Calibration |
| `--max_num_instances_per_eval_task` | Maximum number of test instances per task |
| `--output_dir` | Where results are saved |

### Demonstration budget experiment

To reproduce the k = 4 experiment, set `--num_pos_examples 4` and change the output directory, for example:

```bash
python -m src.superni.run_completions_eval \
    --model Qwen/Qwen2.5-32B-Instruct \
    --data_dir data/eval/superni/splits/classification_tasks/ \
    --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples 4 \
    --eval_bias_score --eval_looc --eval_cc --eval_dc \
    --max_num_instances_per_eval_task 100 \
    --output_dir runs/Qwen2.5-32B-Instruct/4_shots/
```

## Models

| Family | Models |
|---|---|
| Llama | `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.2-3B-Instruct` |
| Phi | `microsoft/Phi-3.5-mini-instruct`, `microsoft/Phi-3.5-MoE-instruct` |
| Mixtral | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Qwen | `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-32B-Instruct` |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |

All experiments were run on four NVIDIA A100 80GB GPUs, using about 110 GPU-hours in total.

## Metrics

- **Macro-F1**: average F1 across labels, giving each label equal weight.
- **RSD**: relative standard deviation of per-label accuracy. Lower means more even performance across labels.
- **BiasScore**: distance between the predicted label distribution and a uniform distribution (Reif and Schwartz, 2024). Its maximum is 1 - 1/K, so compare it only within the same label-space group.

## Citation

```bibtex
@inproceedings{anonymous2027labelbias,
  title     = {Label Bias Calibration in LLMs Depends on Label-Space Size},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```

## Acknowledgements

This code builds on the evaluation suite of Reif and Schwartz (2024). Super-NaturalInstructions and Hugging Face Transformers are released under the Apache 2.0 licence. Models are used under their respective licences. This repository does not redistribute data or model weights.
