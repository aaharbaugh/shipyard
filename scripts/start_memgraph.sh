#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-shipyard-memgraph}"
IMAGE="${MEMGRAPH_IMAGE:-memgraph/memgraph:latest}"
BOLT_PORT="${MEMGRAPH_BOLT_PORT:-7687}"
LAB_PORT="${MEMGRAPH_LAB_PORT:-7444}"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Memgraph is already running as $CONTAINER_NAME"
  docker ps --filter "name=$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Starting existing Memgraph container: $CONTAINER_NAME"
  docker start "$CONTAINER_NAME" >/dev/null
else
  echo "Creating and starting Memgraph container: $CONTAINER_NAME"
  docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${BOLT_PORT}:7687" \
    -p "${LAB_PORT}:7444" \
    "$IMAGE" >/dev/null
fi

echo
docker ps --filter "name=$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo
echo "Next step:"
echo "  $ROOT_DIR/scripts/index_graph.sh"
