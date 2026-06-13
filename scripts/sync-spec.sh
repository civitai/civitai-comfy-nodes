#!/usr/bin/env bash
# Sync the consumer OpenAPI spec from the orchestration repo and regenerate all nodes.
# Run from the repo root after rebuilding Civitai.Orchestration.Api.
set -euo pipefail

cd "$(dirname "$0")/.."

SPEC_SOURCE="${SPEC_SOURCE:-../../civitai-orchestration/repo/src/Civitai.Orchestration.Api/wwwroot/openapi/v2-consumers.json}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -f "$SPEC_SOURCE" ]]; then
    echo "Spec not found at $SPEC_SOURCE — build Orchestration.Api first or set SPEC_SOURCE" >&2
    exit 1
fi

cp "$SPEC_SOURCE" spec/v2-consumers.json
"$PYTHON" -m codegen.generate
"$PYTHON" -m pytest tests -q

echo
echo "Review the generated diff before committing:"
git status --short spec/ civitai_comfy_nodes/generated/
