#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/jqtao/.conda/envs/pytorch2_8/bin/python}"

echo "Python: ${PYTHON_BIN}"
echo "Started at: $(date '+%F %T')"

"${PYTHON_BIN}" scripts/aggregate_downstream.py
"${PYTHON_BIN}" scripts/prepare_scale_scores.py
"${PYTHON_BIN}" scripts/analyze_correlations.py

echo "Finished at: $(date '+%F %T')"
