"""Ensure repo root is on sys.path so `app.*` imports work under Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_root() -> Path:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _REPO_ROOT


ensure_repo_root()
