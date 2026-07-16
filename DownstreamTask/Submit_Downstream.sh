#!/bin/bash
#SBATCH --job-name=T23-SDown-NA-R1to5
#SBATCH --partition=gpu
#SBATCH --nodelist=ws4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=32-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p logs outputs outputs/figures prompts data

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/jqtao/.conda/envs/pytorch2_8

export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/jqtao/.conda/envs/pytorch2_8/bin/python}"

export DOWNSTREAM_REPEAT_START="${DOWNSTREAM_REPEAT_START:-1}"
export DOWNSTREAM_N_REPEATS="${DOWNSTREAM_N_REPEATS:-5}"
export DOWNSTREAM_LANGUAGES="${DOWNSTREAM_LANGUAGES:-zh en}"
export DOWNSTREAM_TIMEOUT="${DOWNSTREAM_TIMEOUT:-90}"
export DOWNSTREAM_MAX_RETRIES="${DOWNSTREAM_MAX_RETRIES:-5}"
export DOWNSTREAM_RETRY_BACKOFF_BASE="${DOWNSTREAM_RETRY_BACKOFF_BASE:-2}"
export DOWNSTREAM_RETRY_BACKOFF_MAX="${DOWNSTREAM_RETRY_BACKOFF_MAX:-20}"
export DOWNSTREAM_OVERWRITE="${DOWNSTREAM_OVERWRITE:-0}"
export DOWNSTREAM_DRY_RUN="${DOWNSTREAM_DRY_RUN:-0}"

echo "Workdir: ${SCRIPT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Started at: $(date '+%F %T')"
echo "Repeat start: ${DOWNSTREAM_REPEAT_START}"
echo "Repeats: ${DOWNSTREAM_N_REPEATS}"
echo "Languages: ${DOWNSTREAM_LANGUAGES}"
echo "Timeout: ${DOWNSTREAM_TIMEOUT}"
echo "Max retries: ${DOWNSTREAM_MAX_RETRIES}"
echo "Retry backoff base: ${DOWNSTREAM_RETRY_BACKOFF_BASE}"
echo "Retry backoff max: ${DOWNSTREAM_RETRY_BACKOFF_MAX}"
echo "Overwrite: ${DOWNSTREAM_OVERWRITE}"
echo "Dry run: ${DOWNSTREAM_DRY_RUN}"
echo "API payload fields: model, messages"

if [[ ! -f "../Test_File/api_config.json" ]] && [[ -z "${N1N_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${API_KEY:-}" ]]; then
  echo "Missing API credentials: set env vars or provide ../Test_File/api_config.json" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

required = ["requests", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("Python dependencies already satisfied.")
PY

"${PYTHON_BIN}" scripts/build_downstream_tasks.py

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

models = json.loads(Path("data/model_list.json").read_text(encoding="utf-8"))
tasks = json.loads(Path("prompts/downstream_tasks.json").read_text(encoding="utf-8"))
enabled = [m["model_name"] if isinstance(m, dict) else m for m in models if not isinstance(m, dict) or m.get("enabled", True)]
print(f"Selected models ({len(enabled)}): {enabled}")
print(f"Downstream tasks ({len(tasks)}): {[task['task_id'] for task in tasks]}")
PY

"${PYTHON_BIN}" scripts/run_downstream.py
"${PYTHON_BIN}" scripts/parse_downstream.py

"${PYTHON_BIN}" - <<'PY'
import json
from collections import Counter
from pathlib import Path

import pandas as pd

root = Path(".")
rows = [
    json.loads(line)
    for line in (root / "outputs/downstream_raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
scores = pd.read_csv(root / "outputs/downstream_scores_long.csv")
repeat_start = int(__import__("os").environ["DOWNSTREAM_REPEAT_START"])
n_repeats = int(__import__("os").environ["DOWNSTREAM_N_REPEATS"])


def count_true(frame, column):
    return int(frame[column].astype(str).str.lower().eq("true").sum()) if not frame.empty else 0


print("Quality summary:")
print(f"  raw_total_all_repeats={len(rows)}")
for repeat_id in range(repeat_start, repeat_start + n_repeats):
    repeat_rows = [r for r in rows if r.get("repeat_id") == repeat_id]
    repeat_scores = scores[scores["repeat_id"].eq(repeat_id)]
    print(f"  raw_repeat_{repeat_id}={len(repeat_rows)}")
    print(f"  errors_repeat_{repeat_id}={sum(1 for r in repeat_rows if r.get('error'))}")
    print(f"  na_repeat_{repeat_id}={count_true(repeat_scores, 'is_na')}")
    print(
        f"  invalid_repeat_{repeat_id}="
        f"{len(repeat_scores) - count_true(repeat_scores, 'response_valid')}"
    )
    print(f"  scored_repeat_{repeat_id}={count_true(repeat_scores, 'score_available')}")
print(f"  by_repeat={dict(sorted(Counter(r.get('repeat_id') for r in rows).items()))}")
PY

echo "Finished at: $(date '+%F %T')"
