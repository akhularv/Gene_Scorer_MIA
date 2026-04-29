"""Small helpers for paths inside this repo."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str) -> str:
    """Resolve a repo-relative or absolute path."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(PROJECT_ROOT / path_obj)
