from __future__ import annotations

from pathlib import Path

from workflow_runtime.artifacts.checks import nonempty_file_complete

_MODEL_PATTERNS = ("*.model", "*.pt", "*.pt2", "graph.pb", "nep.txt", "model.ckpt-*.index")


def model_artifacts_complete(root: str | Path, *, minimum_count: int = 1) -> bool:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return False
    model_paths: list[Path] = []
    for pattern in _MODEL_PATTERNS:
        model_paths.extend(root_path.rglob(pattern))
    complete = [path for path in model_paths if nonempty_file_complete(path)]
    return len(complete) >= int(minimum_count)


def teacher_foundation_train_complete(root: str | Path) -> bool:
    return model_artifacts_complete(Path(root) / "teacher_foundation")


def teacher_committee_finetune_complete(root: str | Path, *, teacher_count: int = 1) -> bool:
    return model_artifacts_complete(root, minimum_count=teacher_count)
