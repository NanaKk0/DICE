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
        deepmd_auto_batch_size: int | str | None = None,
        type_map: list[str] | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.deepmd_auto_batch_size = deepmd_auto_batch_size if deepmd_auto_batch_size is not None else batch_size
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

    def eval_batch(
        self,
        *,
        coords: Any,
        cells: Any,
        atom_types: Any,
        fparam: Any = None,
        aparam: Any = None,
        charge_spin: Any = None,
    ) -> tuple[Any, ...]:
        potential = self._load_potential()
        kwargs: dict[str, Any] = {}
        if fparam is not None:
            kwargs["fparam"] = fparam
        if aparam is not None:
            kwargs["aparam"] = aparam
        if charge_spin is not None:
            kwargs["charge_spin"] = charge_spin
        return potential.eval(coords, cells, atom_types, **kwargs)

    def preflight(self) -> None:
        potential = self._load_potential()
        runtime_type_map = _runtime_type_map(potential)
        if runtime_type_map is not None and self.type_map and list(runtime_type_map) != self.type_map:
            raise RuntimeError(
                "DeePMD teacher Python relabel type_map mismatch: "
                f"model {self.model_path} exposes {list(runtime_type_map)!r}, expected {self.type_map!r}"
            )

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
        if self.deepmd_auto_batch_size is not None and self.device:
            constructor_attempts.append({"auto_batch_size": self.deepmd_auto_batch_size, "device": self.device})
        if self.deepmd_auto_batch_size is not None:
            constructor_attempts.append({"auto_batch_size": self.deepmd_auto_batch_size})
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


def _runtime_type_map(potential: Any) -> list[str] | None:
    for attr in ("get_type_map", "get_type_map_list"):
        candidate = getattr(potential, attr, None)
        if callable(candidate):
            value = candidate()
            if value is not None:
                return [str(item) for item in value]
    for attr in ("type_map", "type_map_list"):
        value = getattr(potential, attr, None)
        if value is not None:
            return [str(item) for item in value]
    return None


def build_deepmd_evaluator(
    model_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    deepmd_auto_batch_size: int | str | None = None,
    type_map: list[str] | None = None,
) -> DeepMDPythonEvaluator:
    return DeepMDPythonEvaluator(
        model_path,
        device=device,
        batch_size=batch_size,
        deepmd_auto_batch_size=deepmd_auto_batch_size,
        type_map=type_map,
    )
