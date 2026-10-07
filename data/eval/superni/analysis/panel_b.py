import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ========= Config =========
HERE = Path.cwd()
ROOT = next((p for p in [HERE, *HERE.parents] if (p / "runs_stale").is_dir()), None)
if ROOT is None:
    raise SystemExit("runs_stale not found in any parent of " + str(HERE))

INFO_PATH = next(ROOT.rglob("test_tasks_info.csv"))
RUNS_DIR  = ROOT / "runs_stale"
ALPHA     = 0.05
K_LO, K_HI = 3, 5                     # panel (b)

ORDER = ["Llama-3.2-3B", "Llama-3.2-3B-Instruct",
         "Phi-3.5-mini-instruct", "Phi-3.5-MoE-instruct",
         "Mixtral-8x7B-v0.1", "Mixtral-8x7B-Instruct-v0.1",
         "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
         "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-32B"]
GAPS = {"Phi-3.5-mini-instruct", "Mixtral-8x7B-v0.1",
        "Qwen2.5-32B", "DeepSeek-R1-Distill-Llama-8B"}
PREF   = ["", "cc_", "dc_", "looc_"]          # Ori., CC, DCC, LOOC
LOWER  = {"rsd", "bias_score"}

# ========= Load =========
df = pd.read_csv(INFO_PATH)
frames = []
for p in sorted(RUNS_DIR.glob("*/8_shots/task_metrics.csv")):
    f = pd.read_csv(p)
    f["Model"] = p.parent.parent.name
    frames.append(f)
if not frames:
    raise SystemExit(f"Nothing matched {RUNS_DIR}/*/8_shots/task_metrics.csv")
res = pd.concat(frames, ignore_index=True)

# ========= Helpers =========
def parse_dist(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return None

def task_id(name):
    m = re.match(r"(task\d+)", str(name))
    return m.group(1) if m else None

# ========= Compute =========
dfv = df[df["Labels Distribution"].apply(parse_dist).notna()].copy()
dfv["tid"] = dfv["Name"].apply(task_id)
dfv["K"]   = dfv["# Answer Choices"].astype(int)

res["tid"] = res.iloc[:, 0].apply(task_id)
res = res.merge(dfv[["tid", "K"]], on="tid", how="inner")

sub = res[(res.K >= K_LO) & (res.K <= K_HI)]
n = sub.tid.nunique()
print(f"% panel (b): {n} tasks, {sub.Model.nunique()} models\n")
print(r"\multicolumn{13}{l}{\textit{(b) Intermediate label spaces ($3 \le K \le 5$), $n = 60$.}} \\")
print(r"\midrule")

wins = {"Ori.": 0, "CC": 0, "DCC": 0, "LOOC": 0}
for m in ORDER:
    d = sub[sub.Model == m]
    if d.empty:
        print(f"% {m}: no rows")
        continue
    cells = []
    for metric in ["macro_f1", "rsd", "bias_score"]:
        cols = [pre + metric if pre else metric for pre in PREF]
        vals = [d[c].mean() * 100 for c in cols]
        rank = sorted(range(4), key=lambda i: vals[i],
                      reverse=metric not in LOWER)
        win, second = rank[0], rank[1]
        if metric == "macro_f1":
            wins[["Ori.", "CC", "DCC", "LOOC"][win]] += 1
        try:
            _, pv = stats.wilcoxon(d[cols[win]], d[cols[second]])
        except ValueError:
            pv = 1.0
        for i, v in enumerate(vals):
            txt = f"{v:.1f}"
            if abs(v - vals[win]) < 0.05:
                txt = r"\bfseries " + txt
                if pv >= ALPHA:
                    txt += r"\rlap{$^{\dag}$}"
            cells.append(txt)
    pre = "\\addlinespace[2pt]\n" if m in GAPS else ""
    print(f"{pre}{m}\n& " + " & ".join(cells[0:4])
          + "\n& " + " & ".join(cells[4:8])
          + "\n& " + " & ".join(cells[8:12]) + r" \\")

print(f"\n% Macro-F1 winners in panel (b): {wins}")