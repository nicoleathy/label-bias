# Label Bias Calibration in LLMs: Choosing Calibration by Label-Space Size

Code and results for the paper *Label Bias Calibration in LLMs: Choosing Calibration by Label-Space Size*.

We compare three label-agnostic calibration methods for in-context classification:

- **CC**: Contextual Calibration (Zhao et al., 2021), which estimates the label prior from a content-free input such as `N/A`.
- **DCC**: Domain Contextual Calibration (Fei et al., 2023), which estimates it from random in-domain text.
- **LOOC**: Leave-One-Out Calibration (Reif and Schwartz, 2024), which estimates it from the in-context demonstrations themselves.

We run all three on 279 Super-NaturalInstructions classification tasks with 10 LLMs from 5 families. LOOC is best on average, but its advantage depends on the number of labels K. LOOC can only see labels that appear among its k demonstrations, so with k = 8 it wins for K ≤ 5, while CC or DCC win for K ≥ 6.

**Selection rule:** use LOOC when the demonstrations reliably cover the label space (K ≤ 5 with k = 8), and CC or DCC otherwise.

## Repository structure

```
.
├── src/superni/
│   ├── run_completions_eval.py    # main entry point: builds prompts, scores answers, runs CC/DCC/LOOC/BiasScore
│   ├── compute_metrics.py         # Macro-F1, RSD and BiasScore before and after calibration
│   ├── task_info/full_task_info.csv
│   └── utils/
│       ├── ni_dataset.py          # Super-NI dataset loader
│       ├── ni_collator.py         # prompt construction (instruction + k demonstrations + input)
│       ├── data_utils.py          # builds CC, DCC, LOOC and BiasScore inputs
│       ├── predict_utils.py       # conditional completion scoring with a shared-prefix cache
│       ├── model_utils.py         # Hugging Face model and tokenizer loading
│       └── metrics.py             # RSD and BiasScore
├── scripts/
│   ├── prepare_superni_data.sh    # downloads Super-NI and builds the 279-task split
│   ├── extract_classification_tasks.py
│   └── run_eval.sh                # runs all methods for one or more models
├── analysis/
│   ├── make_tables.py             # Tables 1-4, LOOC margins and label coverage
│   └── make_figures.py            # Figures 1 and 2
├── data/classification_tasks/
│   ├── test_tasks.txt             # the 279 task IDs
│   └── test_tasks_info.csv        # label-space size and label distribution per task
└── results/
    ├── k8/<model>/                # per-task and mean metrics, k = 8 (Tables 1, 2 and 4)
    └── k4/Qwen2.5-32B-Instruct/   # k = 4 ablation (Table 3)
```

## Reproducing the tables and figures (no GPU needed)

The per-task metrics for every model are in `results/`, so the analysis runs on a laptop:

```bash
pip install numpy pandas scipy matplotlib
python analysis/make_tables.py     # prints Tables 1-4, LOOC margins and label coverage
python analysis/make_figures.py    # writes figures/ (PDF, PNG and CSV)
```

Each `results/<k>/<model>/task_metrics.csv` has one row per task and these columns, with values as fractions:

| Prefix | Setting |
|---|---|
| none | uncalibrated (Ori.) |
| `cc_` | Contextual Calibration |
| `dc_` | Domain Contextual Calibration |
| `looc_` | Leave-One-Out Calibration |

Each prefix is followed by `macro_f1`, `rsd` or `bias_score`. `mean_metrics.json` holds the means over all 279 tasks.

## Running the full pipeline

### 1. Environment

Tested with Python 3.11, PyTorch 2.4 and Transformers 4.46.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

```bash
bash scripts/prepare_superni_data.sh
```

This downloads [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) into `data/eval/superni/` and runs `scripts/extract_classification_tasks.py`. The script keeps English, single-label classification tasks with fewer than 50 classes and at least 396 instances, which gives the 279 tasks listed in `data/classification_tasks/test_tasks.txt`. For each task it holds out a stratified pool of 96 examples (seed 42) to draw demonstrations from, so test inputs are never used as demonstrations.

### 3. Evaluation

Some models (such as Llama) are gated on Hugging Face. Set your own token in the environment and do not write it into any file:

```bash
export HF_TOKEN=<your token>

bash scripts/run_eval.sh                                   # all 10 models, k = 8
bash scripts/run_eval.sh Qwen/Qwen2.5-32B-Instruct         # a single model
K=4 bash scripts/run_eval.sh Qwen/Qwen2.5-32B-Instruct     # k = 4 ablation (Table 3)
```

Each run writes to `runs/<model>/<k>_shots/`:

- `full_outputs.pickle`: per-instance predictions, probabilities and gold labels (uncalibrated)
- `cc/`, `dc/`, `looc/`, `bias_score/`: the same outputs for the calibration and BiasScore inputs
- `prompts_info.pickle`: the demonstrations used for each task
- `task_metrics.csv`, `mean_metrics.json`: the final metrics, in the same format as `results/`

To recompute the metrics for an existing run directory:

```bash
python -m src.superni.compute_metrics --output_dir runs/<model>/8_shots/
```

Main options of `src.superni.run_completions_eval`:

| Argument | Default | Meaning |
|---|---|---|
| `--model` | | Hugging Face model ID |
| `--num_pos_examples` | 0 | number of demonstrations k (8 in the paper) |
| `--max_num_instances_per_eval_task` | 1000 | test instances per task (100 in the paper) |
| `--max_source_length` | 2000 | token budget for the prompt |
| `--eval_cc`, `--eval_dc`, `--eval_looc` | off | run each calibration method |
| `--eval_bias_score` | off | compute BiasScore on held-out instances |
| `--seed` | 42 | random seed |

## How it works

1. **Prompt construction.** Each prompt has the task instruction, k demonstrations from the held-out pool and the test input as `Input: ... Output:`.
2. **Completion scoring.** For each of the K answer choices we append its verbaliser to the prompt and compute the cross-entropy over the answer tokens only. The negative loss is the score, and the K scores are normalised into probabilities. The shared prompt is encoded once and its key-value cache is reused across choices.
3. **Calibration.** Each method gets a label prior from inputs that carry no real answer, divides the test probabilities by it and renormalises. CC and DCC need one extra pass per prompt, O(nc). LOOC repeats this for each of the k leave-one-out contexts, O(knc).
4. **Metrics.** Macro-F1, RSD (standard deviation of per-label accuracy divided by mean accuracy) and BiasScore (distance of the mean predicted label distribution from uniform, with maximum 1 − 1/K).

## Compute

The experiments used about 110 GPU-hours on four NVIDIA A100 80GB GPUs. Models are loaded in bfloat16 with `device_map="auto"`.

## Acknowledgements

This code builds on the public implementation of Reif and Schwartz (2024), *Beyond Performance: Quantifying and Mitigating Label Bias in LLMs* (NAACL 2024), and uses Super-NaturalInstructions (Wang et al., 2022) and Hugging Face Transformers, both released under Apache 2.0.

## Citation

Citation details will be added after the anonymity period.
