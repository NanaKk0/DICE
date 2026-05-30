from __future__ import annotations

from pathlib import Path
from typing import Any


class DeepMDPythonEvaluator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int | None = None,
        model_family: str = "dpa4",
        type_map: list[str] | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.batch_size = batch_size
        self.model_family = model_family
        self.type_map = [str(item) for item in type_map or []]
        self._potential = None

    def evaluate(self, atoms: Any) -> dict[str, Any]:
        if not self.type_map:
            raise ValueError("DeePMD teacher Python evaluator requires type_map for atom-type mapping")
        potential = self._load_potential()
        import numpy as np

        symbols = list(atoms.get_chemical_symbols())
        type_index = {element: index for index, element in enumerate(self.type_map)}
        try:
            atom_types = np.asarray([type_index[symbol] for symbol in symbols], dtype=np.int32)
        except KeyError as exc:
            raise ValueError(f"Element {exc.args[0]!r} is not in DeePMD teacher type_map {self.type_map!r}") from exc

        coords = np.asarray(atoms.get_positions(), dtype=float).reshape(1, -1)
        cell = atoms.get_cell()
        box = None if getattr(cell, "volume", 0.0) < 1e-12 else np.asarray(cell.array, dtype=float).reshape(1, 9)
        result = potential.eval(coords, box, atom_types)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("DeepMD evaluator returned an unsupported result; expected at least energy and forces")

        energy, forces = result[0], result[1]
        payload = {
            "energy": float(np.asarray(energy).reshape(-1)[0]),
            "forces": np.asarray(forces, dtype=float).reshape(len(atoms), 3),
        }
        if len(result) >= 3 and result[2] is not None and box is not None:
            virial = np.asarray(result[2], dtype=float).reshape(3, 3)
            volume = float(abs(cell.volume))
            if volume > 1e-12:
                payload["stress"] = -virial / volume
        return payload

    def _load_potential(self) -> Any:
        if self._potential is not None:
            return self._potential
        try:
            from deepmd.infer import DeepPot
        except Exception as exc:
            raise RuntimeError(
                "DeePMD teacher Python relabel requires DeePMD-kit with deepmd.infer.DeepPot available. "
                "Run this in the DPA4/DeePMD environment or inject a test evaluator."
            ) from exc

        constructor_attempts: list[dict[str, Any]] = []
        if self.batch_size is not None and self.device:
            constructor_attempts.append({"auto_batch_size": self.batch_size, "device": self.device})
        if self.batch_size is not None:
            constructor_attempts.append({"auto_batch_size": self.batch_size})
        if self.device:
            constructor_attempts.append({"device": self.device})
        constructor_attempts.append({})

        errors: list[str] = []
        for kwargs in constructor_attempts:
            try:
                self._potential = DeepPot(self.model_path, **kwargs)
                return self._potential
            except TypeError as exc:
                errors.append(str(exc))
                continue
        raise RuntimeError(
            "Unable to construct DeePMD DeepPot evaluator with supported constructor variants: "
            + " | ".join(errors)
        )


def build_deepmd_evaluator(
    model_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    model_family: str = "dpa4",
    type_map: list[str] | None = None,
) -> DeepMDPythonEvaluator:
    return DeepMDPythonEvaluator(
        model_path,
        device=device,
        batch_size=batch_size,
        model_family=model_family,
        type_map=type_map,
    )
