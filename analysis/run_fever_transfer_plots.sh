#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PY="${PY:-${ROOT_DIR}/.venv312/bin/python}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT_DIR}/results/analysis/transfer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ANALYSIS_DIR}/plots}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"

export MPLCONFIGDIR

"${PY}" "${SCRIPT_DIR}/plot_fever_paraphrase_results.py" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --formats png pdf \
  --dpi 220 \
  --language en \
  --style paper \
  --sort-gain desc \
  --with-regression-line
