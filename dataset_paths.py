from __future__ import annotations

from pathlib import Path


DATASET_DIR = Path("dataset")


def dataset_path(name: str) -> Path:
    return DATASET_DIR / name


def resolve_dataset_path(path: Path) -> Path:
    """Resolve old root-level data_* paths after moving them under dataset/."""
    if path.exists() or path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0].startswith("data_"):
        candidate = DATASET_DIR.joinpath(*parts)
        if candidate.exists():
            return candidate

        raw_candidate = DATASET_DIR.joinpath("raw", *parts)
        if raw_candidate.exists():
            return raw_candidate

    return path
