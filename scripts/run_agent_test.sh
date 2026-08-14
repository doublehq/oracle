#!/usr/bin/env bash
# Headless agent driver for the SP500 direct-index e2e test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPT_FILE="${ROOT}/tests/agent/e2e_sp500_direct_index.md"
LOG_DIR="${ROOT}/artifacts"
mkdir -p "$LOG_DIR"

if ! command -v cursor-agent >/dev/null 2>&1; then
  echo "cursor-agent not found on PATH" >&2
  exit 1
fi

PROMPT="$(sed -n '/^## Prompt$/,$p' "$PROMPT_FILE" | tail -n +2)"

echo "=== Running cursor-agent e2e test ==="
rm -f "$LOG_DIR/e2e_result.json" "$LOG_DIR/rh_snapshot.json"
cursor-agent -p --workspace "$ROOT" \
  --output-format stream-json \
  "$PROMPT" 2>&1 | tee "$LOG_DIR/agent_e2e.log"

echo ""
echo "=== Running deterministic verifier ==="
python scripts/verify_e2e_run.py \
  --result "$LOG_DIR/e2e_result.json" \
  --snapshot "$LOG_DIR/rh_snapshot.json"
