#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.shipyard/data/graph/index}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/.shipyard/data/graph/runtime}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RUN_DIR"

echo "Indexing Shipyard into: $OUTPUT_DIR"
(
  cd "$RUN_DIR"
  env -u OPENAI_API_KEY -u OPENAI_MODEL \
    "$ROOT_DIR/.venv/bin/cgr" index \
      --repo-path "$ROOT_DIR" \
      --output-proto-dir "$OUTPUT_DIR"
)

echo
echo "Graph stats:"
(
  cd "$RUN_DIR"
  env -u OPENAI_API_KEY -u OPENAI_MODEL \
    "$ROOT_DIR/.venv/bin/cgr" stats
)
echo
echo "API graph status:"
curl -s http://127.0.0.1:8000/graph/status || true
echo
