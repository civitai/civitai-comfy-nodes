#!/usr/bin/env bash
# Sync the consumer OpenAPI spec and regenerate all nodes.
# Source resolution: the local orchestration build if present (run from a full dev-stack
# checkout after rebuilding Civitai.Orchestration.Api), otherwise the published spec over
# HTTP (the path CI takes, since it has no orchestration checkout). Override either with
# SPEC_SOURCE / SPEC_URL.
set -euo pipefail

cd "$(dirname "$0")/.."

SPEC_SOURCE="${SPEC_SOURCE:-../../civitai-orchestration/repo/src/Civitai.Orchestration.Api/wwwroot/openapi/v2-consumers.json}"
SPEC_URL="${SPEC_URL:-https://orchestration.civitai.com/openapi/v2-consumers.json}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ -n "$SPEC_SOURCE" && -f "$SPEC_SOURCE" ]]; then
    echo "Using local spec: $SPEC_SOURCE"
    cp "$SPEC_SOURCE" spec/v2-consumers.json
else
    echo "Local spec not found; fetching $SPEC_URL"
    curl -fsSL "$SPEC_URL" -o spec/v2-consumers.json
fi
"$PYTHON" -m codegen.generate
"$PYTHON" -m pytest tests -q

echo
echo "Review the generated diff before committing:"
git status --short spec/ civitai_comfy_nodes/generated/
