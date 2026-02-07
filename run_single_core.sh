#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CPU_CORE="${1:-2}"
METADATA="${2:-data/metadata.json}"

RESULTS_DIR="results"
CSV_OUT="$RESULTS_DIR/results_single_core.csv"
SUMMARY_OUT="$RESULTS_DIR/summary_single_core.json"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

rotate_if_exists() {
  local target="$1"
  if [[ -f "$target" ]]; then
    local backup="${target%.csv}_${TIMESTAMP}.csv"
    if [[ "$target" == *.json ]]; then
      backup="${target%.json}_${TIMESTAMP}.json"
    fi
    mv "$target" "$backup"
    echo "[INFO] Existing output moved to: $backup"
  fi
}

rotate_if_exists "$CSV_OUT"
rotate_if_exists "$SUMMARY_OUT"

echo "[INFO] MODE: single-core (comparative baseline)"
python3 runner/run_all.py \
  --run-mode single-core \
  --metadata "$METADATA" \
  --warmup 5 --runs 30 --repeat 50 \
  --profile portable \
  --stability-enable --stability-mode wait --stability-timeout-sec 60 \
  --cpu-util-max 20 --disk-io-mbps-max 5 --mem-available-min-mb 2048 \
  --enforce-single-thread --cpu-affinity "$CPU_CORE" \
  --variants naive,simd,opt,blas,par \
  --baseline python-naive \
  --impls python-naive,python-numpy,python-vectorized,c-naive,c-simd-portable,cpp-naive,cpp-simd-portable,rust-naive,rust-simd,rust-par,go-naive,go-opt,java-naive,java-opt \
  --results-csv "$CSV_OUT" \
  --summary-json "$SUMMARY_OUT"

echo "[INFO] Wrote: $CSV_OUT"
echo "[INFO] Wrote: $SUMMARY_OUT"
