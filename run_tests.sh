#!/bin/bash
# Submit multiple SGC test jobs in parallel
# Usage: ./run_tests.sh [-p PROFILE] [--watch]

PROFILE="${SGCLI_PROFILE:-SGCCLI}"
WATCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -p) PROFILE="$2"; shift 2 ;;
    --watch) WATCH="--watch"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

JOBS=(
  # H100 tests
  "SGC_hello_world/train_workload.yaml"
  "SGC_hello_world_error/train.yaml"
  # A10 tests
  "SGC_hello_world_a10/train_workload.yaml"
  "SGC_hello_world_error_a10/train.yaml"
)

echo "=== Submitting ${#JOBS[@]} test jobs (profile: $PROFILE) ==="

for job in "${JOBS[@]}"; do
  echo ""
  echo ">>> Submitting: $job"
  sgcli run -f "$SCRIPT_DIR/$job" -p "$PROFILE" $WATCH &
done

echo ""
echo "=== All jobs submitted. Waiting for completion... ==="
wait
echo "=== Done ==="
