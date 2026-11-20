# Label Bias Calibration in LLMs: Choosing Calibration by Label-Space Size

Code and results for the paper *Label Bias Calibration in LLMs Depends on Label-Space Size*.

We compare three label-agnostic calibration methods for in-context classification:

- **CC**: Contextual Calibration (Zhao et al., 2021), which estimates the label prior from a content-free input such as `N/A`.
- **DCC**: Domain Contextual Calibration (Fei et al., 2023), which estimates it from random in-domain text.
- **LOOC**: Leave-One-Out Calibration (Reif and Schwartz, 2024), which estimates it from the in-context demonstrations themselves.

We run all three on 279 Super-NaturalInstructions classification tasks with 10 LLMs from 5 families. LOOC is best on average, but its advantage depends on the number of labels K. LOOC can only see labels that appear among its k demonstrations, so with k = 8 it wins for K ≤ 5, while CC or DCC win for K ≥ 6.

**Selection rule:** use LOOC when the demonstrations reliably cover the label space (K ≤ 5 with k = 8), and CC or DCC otherwise.

## Structure

```
src/superni/       evaluation pipeline (prompting, scoring, calibration, metrics)
scripts/           data preparation and evaluation scripts
analysis/          scripts for the paper's tables and figures
data/              the 279 task IDs and their label distributions
results/           per-task metrics: k8/<model>/ (main results), k4/ (k = 4 ablation)
```

## Reproduce the tables and figures (no GPU)

```bash
pip install numpy pandas scipy matplotlib
python analysis/make_tables.py     # Tables 1-4 and label coverage
python analysis/make_figures.py    # Figures 1 and 2, written to figures/
```

## Run the full pipeline

Requires Python 3.11, PyTorch 2.4 and Transformers 4.46.

```bash
pip install -r requirements.txt
bash scripts/prepare_superni_data.sh       # download Super-NI, build the 279-task split

export HF_TOKEN=<your token>               # for gated models such as Llama
bash scripts/run_eval.sh                   # all 10 models, k = 8
K=4 bash scripts/run_eval.sh Qwen/Qwen2.5-32B-Instruct   # k = 4 ablation
```

Results are written to `runs/<model>/<k>_shots/`. The experiments used about 110 GPU-hours on four A100 80GB GPUs.

## Acknowledgements

Builds on the code of Reif and Schwartz (2024) and uses Super-NaturalInstructions and Hugging Face Transformers (Apache 2.0).
