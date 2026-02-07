#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

THREADS="${1:-$(nproc)}"
METADATA="${2:-data/metadata.json}"
CPU_SET_MULTI="${3:-}"

if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
  echo "[FAIL] THREADS must be an integer >= 2" >&2
  exit 1
fi
if (( THREADS < 2 )); then
  echo "[FAIL] multi-core mode requires THREADS >= 2" >&2
  exit 1
fi
if [[ -z "$CPU_SET_MULTI" ]]; then
  CPU_SET_MULTI="0-$((THREADS-1))"
fi

RESULTS_DIR="results"
CSV_OUT="$RESULTS_DIR/results_multi_core.csv"
SUMMARY_OUT="$RESULTS_DIR/summary_multi_core.json"
CSV_SCALABLE_OUT="$RESULTS_DIR/results_multi_core_scalable.csv"
SUMMARY_SCALABLE_OUT="$RESULTS_DIR/summary_multi_core_scalable.json"
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
rotate_if_exists "$CSV_SCALABLE_OUT"
rotate_if_exists "$SUMMARY_SCALABLE_OUT"

# Explicit thread configuration for runtime libraries.
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export GOMAXPROCS="$THREADS"
export RAYON_NUM_THREADS="$THREADS"

echo "[INFO] MODE: multi-core (throughput / scalability)"
echo "[INFO] Threads T=$THREADS"
echo "[INFO] CPU set=$CPU_SET_MULTI"
python3 runner/run_all.py \
  --run-mode multi-core \
  --threads "$THREADS" \
  --multicore-scope all \
  --metadata "$METADATA" \
  --warmup 5 --runs 30 --repeat 50 \
  --profile native \
  --stability-enable --stability-mode wait --stability-timeout-sec 60 \
  --cpu-util-max 20 --disk-io-mbps-max 5 --mem-available-min-mb 2048 \
  --cpu-set "$CPU_SET_MULTI" \
  --variants simd,opt,blas,par \
  --baseline python-naive \
  --impls python-naive,python-numpy,python-vectorized,c-simd-native,cpp-simd-native,rust-simd,rust-par,go-opt,java-opt \
  --results-csv "$CSV_OUT" \
  --summary-json "$SUMMARY_OUT"

echo "[INFO] MODE: multi-core scalable-only (throughput)"
python3 runner/run_all.py \
  --run-mode multi-core \
  --threads "$THREADS" \
  --multicore-scope scalable-only \
  --metadata "$METADATA" \
  --warmup 5 --runs 30 --repeat 50 \
  --profile native \
  --stability-enable --stability-mode wait --stability-timeout-sec 60 \
  --cpu-util-max 20 --disk-io-mbps-max 5 --mem-available-min-mb 2048 \
  --cpu-set "$CPU_SET_MULTI" \
  --variants simd,opt,blas,par \
  --baseline python-naive \
  --impls python-naive,python-numpy,python-vectorized,c-simd-native,cpp-simd-native,rust-simd,rust-par,go-opt,java-opt \
  --results-csv "$CSV_SCALABLE_OUT" \
  --summary-json "$SUMMARY_SCALABLE_OUT"

echo "[INFO] Wrote: $CSV_OUT"
echo "[INFO] Wrote: $SUMMARY_OUT"
echo "[INFO] Wrote: $CSV_SCALABLE_OUT"
echo "[INFO] Wrote: $SUMMARY_SCALABLE_OUT"
