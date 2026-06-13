"""Golden-file test for the emitter. On intentional changes, regenerate via:
UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_emit_golden.py
"""

import json
import os
from pathlib import Path

from codegen import emit, generate

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

GOLDEN_CLASSES = ["CivitaiEcho", "CivitaiImageUpscaler", "CivitaiTextToImage", "CivitaiImageGenFlux1Kontext"]


def test_emitted_source_matches_golden():
    spec = json.loads((REPO_ROOT / "spec" / "v2-consumers.json").read_text())
    overrides = json.loads((REPO_ROOT / "codegen" / "overrides.json").read_text())
    nodes = {n.class_name: n for n in generate.build_nodes(spec, overrides)}

    for class_name in GOLDEN_CLASSES:
        source = emit.emit_node(nodes[class_name]) + "\n"
        golden_path = GOLDEN_DIR / f"{class_name}.py.golden"
        if os.environ.get("UPDATE_GOLDEN"):
            golden_path.write_text(source)
            continue
        expected = golden_path.read_text()
        assert source == expected, (
            f"{class_name} emit drifted from golden. If intentional (spec sync or emitter change), "
            "re-run with UPDATE_GOLDEN=1 and review the golden diff."
        )
