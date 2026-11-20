#!/usr/bin/env bash
# Download Super-NaturalInstructions and build the 279-task classification split.
# Run from the repository root.
set -euo pipefail

mkdir -p data/downloads data/eval/superni
wget -O data/downloads/superni_data.zip \
  https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
unzip -q data/downloads/superni_data.zip -d data/downloads/superni
cp -r data/downloads/superni/natural-instructions-master/. data/eval/superni/
rm -r data/downloads

# Writes data/eval/superni/splits/classification_tasks/test_tasks.txt (task list)
# and data/eval/superni/classification_tasks/*.json (edited task files).
python scripts/extract_classification_tasks.py

# The task list should match the one used in the paper.
diff <(sort data/eval/superni/splits/classification_tasks/test_tasks.txt | sed '/^$/d') \
     <(sort data/classification_tasks/test_tasks.txt | sed '/^$/d') \
  && echo "Task list matches data/classification_tasks/test_tasks.txt (279 tasks)."
