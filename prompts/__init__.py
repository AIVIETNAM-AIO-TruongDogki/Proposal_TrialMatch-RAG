"""Eligibility-reasoning prompts (Phase 8) live here as plain .txt files, so a
diff on a prompt is a diff on exactly the prompt. Extraction/HyDE prompts are
inlined as constants in the modules that use them (src/extraction/{schema,query}.py).
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
