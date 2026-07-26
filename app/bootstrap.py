"""Ensure repo root is on sys.path for `app.*` imports."""

from app._path import ensure_repo_root

__all__ = ["ensure_repo_root"]
