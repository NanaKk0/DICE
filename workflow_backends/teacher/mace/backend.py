from __future__ import annotations

from pathlib import Path


def build_mace_calculator_from_declared_models(
    model_path: str | Path | None = None,
    *,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    device: str = "cpu",
    default_dtype: str = "float32",
):
    """Load explicit native MACE model paths for the formal label worker."""
    from mace.calculators import MACECalculator

    if model_paths is not None:
        if not model_paths:
            raise ValueError("MACE calculator model_paths must be non-empty")
        return MACECalculator(
            model_paths=[str(Path(path).resolve()) for path in model_paths],
            device=device,
            default_dtype=default_dtype,
        )
    if model_path is None:
        raise ValueError("MACE calculator requires model_path or model_paths")
    return MACECalculator(
        model_path=str(Path(model_path).resolve()),
        device=device,
        default_dtype=default_dtype,
    )
