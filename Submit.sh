#!/bin/bash
#SBATCH --job-name=T23-API-Main
#SBATCH --partition=gpu
#SBATCH --nodelist=ws4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=32-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p logs

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/jqtao/.conda/envs/pytorch2_8

export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/jqtao/.conda/envs/pytorch2_8/bin/python}"

MODELS_FILE="${MODELS_FILE:-Test_File/API_Selected_Final9.txt}"
SHEETS_FILE="${SHEETS_FILE:-Test_File/Sheet_Selected_Final7.txt}"
PROMPT_VARIANT="${PROMPT_VARIANT:-B}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
  RESULTS_ROOT="Outputs/Results_9API_7Scale_Prompt${PROMPT_VARIANT}"
fi
REPEAT_START="${REPEAT_START:-1}"
REPEATS="${REPEATS:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:-32}"
RETRY_BACKOFF_BASE="${RETRY_BACKOFF_BASE:-2}"
RETRY_BACKOFF_MAX="${RETRY_BACKOFF_MAX:-20}"
LANGUAGES="${LANGUAGES:-zh en}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
export MODELS_FILE SHEETS_FILE RESULTS_ROOT PROMPT_VARIANT REPEAT_START REPEATS TIMEOUT_SEC MAX_RETRIES RESPONSE_MAX_TOKENS RETRY_BACKOFF_BASE RETRY_BACKOFF_MAX LANGUAGES SKIP_EXISTING

if [[ "${PROMPT_VARIANT}" != "A" && "${PROMPT_VARIANT}" != "B" ]]; then
  echo "PROMPT_VARIANT must be A or B." >&2
  exit 1
fi

echo "Workdir: ${SCRIPT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Started at: $(date '+%F %T')"
echo "Models file: ${MODELS_FILE}"
echo "Sheets file: ${SHEETS_FILE}"
echo "Results root: ${RESULTS_ROOT}"
echo "Prompt: ${PROMPT_VARIANT}"
echo "Languages: ${LANGUAGES}"
echo "Repeat start: ${REPEAT_START}"
echo "Repeats: ${REPEATS}"
echo "Timeout: ${TIMEOUT_SEC}"
echo "Max retries: ${MAX_RETRIES}"
echo "Response max tokens: ${RESPONSE_MAX_TOKENS}"
echo "Retry backoff base: ${RETRY_BACKOFF_BASE}"
echo "Retry backoff max: ${RETRY_BACKOFF_MAX}"
echo "Skip existing: ${SKIP_EXISTING}"

if [[ ! -f "run_no_memory_batch.py" ]]; then
  echo "Missing run_no_memory_batch.py in ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "Test_File/Scale_16_Q_F.xlsx" ]]; then
  echo "Missing questionnaire file: Test_File/Scale_16_Q_F.xlsx" >&2
  exit 1
fi

if [[ ! -f "${MODELS_FILE}" ]]; then
  echo "Missing models file: ${MODELS_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SHEETS_FILE}" ]]; then
  echo "Missing sheets file: ${SHEETS_FILE}" >&2
  exit 1
fi

if [[ ! -f "Test_File/api_config.json" ]] && [[ -z "${N1N_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${API_KEY:-}" ]]; then
  echo "Missing API credentials: set env vars or provide Test_File/api_config.json" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

required = ["pandas", "requests", "openpyxl"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("Python dependencies already satisfied.")
PY

"${PYTHON_BIN}" - <<'PY'
import os
import re
from pathlib import Path

models_file = Path(os.environ["MODELS_FILE"])
sheets_file = Path(os.environ["SHEETS_FILE"])

def parse_items(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r'"([^"]+)"', line)
        if match:
            items.append(match.group(1))
    return items

models = parse_items(models_file)
sheets = parse_items(sheets_file)
print(f"Selected models ({len(models)}): {models}")
print(f"Selected sheets ({len(sheets)}): {sheets}")
PY

CLI_ARGS=(
  --models-file "${MODELS_FILE}"
  --sheets-file "${SHEETS_FILE}"
  --results-root "${RESULTS_ROOT}"
  --prompt "${PROMPT_VARIANT}"
  --repeat-start "${REPEAT_START}"
  --repeats "${REPEATS}"
  --timeout "${TIMEOUT_SEC}"
  --max-retries "${MAX_RETRIES}"
  --response-max-tokens "${RESPONSE_MAX_TOKENS}"
  --retry-backoff-base "${RETRY_BACKOFF_BASE}"
  --retry-backoff-max "${RETRY_BACKOFF_MAX}"
)

# shellcheck disable=SC2206
LANGUAGE_ARGS=(${LANGUAGES})
CLI_ARGS+=(--languages "${LANGUAGE_ARGS[@]}")

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CLI_ARGS+=(--skip-existing)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CLI_ARGS+=(--dry-run)
fi

echo "CLI args: ${CLI_ARGS[*]}"
"${PYTHON_BIN}" run_no_memory_batch.py "${CLI_ARGS[@]}"
echo "Finished at: $(date '+%F %T')"
