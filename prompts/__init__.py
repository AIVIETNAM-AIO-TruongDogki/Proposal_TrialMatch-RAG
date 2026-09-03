"""Prompt text lives in plain .txt files here, not as Python string constants —
so editing a prompt never means editing code, and a diff on a prompt is a diff
on exactly the prompt.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
