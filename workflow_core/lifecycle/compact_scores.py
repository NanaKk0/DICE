from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import re
from pathlib import Path
import shutil
from typing import Any

import numpy as np

# Historical reader kept for Child G comparison.  The stable receipt identity
# is duplicated here deliberately so this compatibility module has no outward
# dependency; production writers own the canonical constants.
SCHEMA = "dice.compact-gmm-score"
VERSION = 1


class _OrdinalMapping(Mapping[str, Any]):
    def __init__(self, identities: tuple[str, ...], rows: np.ndarray) -> None:
        self._identities = identities
        self._rows = rows

    def __len__(self) -> int:
        return len(self._identities)

    def __iter__(self) -> Iterator[str]:
        return iter(self._identities)

    def __getitem__(self, identity: str) -> Any:
        # Selection asks only for bounded subsets. Full-population consumers use
        # the ordinal iterator below and never build an identity dictionary.
        try:
            ordinal = self._identities.index(identity)
        except ValueError as exc:
            raise KeyError(identity) from exc
        row = self._rows[ordinal]
        return float(row) if np.ndim(row) == 0 else np.asarray(row)

    def align(self, identities: Sequence[str], *, missing: float) -> tuple[np.ndarray, np.ndarray]:
        """Align an ordered subset to a canonical superset without an identity map."""
        values = np.full(len(identities), missing, dtype=np.float64)
        present = np.zeros(len(identities), dtype=np.bool_)
        source_ordinal = 0
        for target_ordinal, identity in enumerate(identities):
            if (
                source_ordinal < len(self._identities)
                and identity == self._identities[source_ordinal]
            ):
                values[target_ordinal] = float(self._rows[source_ordinal])
                present[target_ordinal] = True
                source_ordinal += 1
        if source_ordinal != len(self._identities):
            raise ValueError("compact GMM identities are not an ordered canonical subset")
        return values, present


class CompactScoreView:
    """Validated mmap view over operation-local ordinal GMM results."""

    def __init__(self, receipt, *, ordered_identities, expected_model_digest, expected_reference_digest):
        self._arrays = {}
        try:
            self._initialize(receipt, ordered_identities=ordered_identities,
                             expected_model_digest=expected_model_digest,
                             expected_reference_digest=expected_reference_digest)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        receipt: Mapping[str, object],
        *,
        ordered_identities: Sequence[str],
        expected_model_digest: str,
        expected_reference_digest: str,
    ) -> None:
        identities = tuple(str(item) for item in ordered_identities)
        count = len(identities)
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("version") != VERSION
            or receipt.get("frame_count") != count
            or receipt.get("ordinal_start") != 0
            or receipt.get("ordinal_stop") != count
            or receipt.get("model_digest") != expected_model_digest
            or receipt.get("reference_digest") != expected_reference_digest
        ):
            raise ValueError("compact GMM receipt identity or digest is invalid")
        raw_root = receipt.get("root")
        arrays = receipt.get("arrays")
        schema = receipt.get("descriptor_schema")
        if not isinstance(raw_root, str) or not isinstance(arrays, dict) or not isinstance(schema, dict):
            raise ValueError("compact GMM receipt schema is invalid")
        self.root = Path(raw_root).resolve()
        # The receipt is population-invariant metadata and remains available
        # after the operation-local mmap files are cleaned up.  Formal smoke
        # and provenance consumers use it without retaining result arrays.
        self.receipt = dict(receipt)
        self.identities = identities
        self.descriptor_schema = schema
        self._arrays: dict[str, np.ndarray] = {}
        required = {"probabilities", "statistics"}
        self._elements = receipt.get("elements")
        element_keys = set()
        if self._elements is not None:
            if not isinstance(self._elements, dict):
                raise ValueError("compact GMM element receipt is invalid")
            for index, (element, metadata) in enumerate(self._elements.items()):
                if not isinstance(element, str) or not element or not isinstance(metadata, dict) or set(metadata) != {"means", "presence"}:
                    raise ValueError("compact GMM element array names are invalid")
                match = re.fullmatch(r"element_(\d+)_means", str(metadata["means"]))
                if match is None or metadata["presence"] != f"element_{match[1]}_presence" or element_keys.intersection(metadata.values()):
                    raise ValueError("compact GMM element array names are invalid")
                element_keys.update(metadata.values())
        if not required.issubset(arrays) or not element_keys.issubset(arrays) or set(arrays) - {"probabilities", "statistics", "means"} - element_keys:
            raise ValueError("compact GMM receipt arrays are invalid")
        for name, raw in arrays.items():
            if not isinstance(raw, dict):
                raise ValueError("compact GMM array receipt is invalid")
            filename = raw.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("compact GMM array filename is invalid")
            path = (self.root / filename).resolve()
            if path.parent != self.root or not path.is_file() or _digest(path) != raw.get("sha256"):
                raise ValueError("compact GMM result file digest is invalid")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            self._arrays[name] = array
            if list(array.shape) != raw.get("shape") or array.dtype.str != raw.get("dtype"):
                raise ValueError("compact GMM result dtype or shape is invalid")
            if array.ndim < 1 or array.shape[0] != count:
                raise ValueError("compact GMM ordinal coverage is invalid")
        probabilities = self._arrays["probabilities"]
        if probabilities.shape != (count,) or probabilities.dtype != np.float64:
            raise ValueError("compact GMM probabilities are invalid")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("compact GMM probabilities are outside [0, 1]")
        statistics = self._arrays["statistics"]
        if statistics.shape != (count,) or statistics.dtype != np.float64 or not np.isfinite(statistics).all():
            raise ValueError("compact GMM statistics are invalid")
        means = self.frame_means
        if means is not None and (means.dtype != np.float32 or means.shape != (count, int(schema["width"])) or not np.isfinite(means).all()):
            raise ValueError("compact GMM frame means are invalid")
        for metadata in (self._elements or {}).values():
            means, presence = self._arrays[metadata["means"]], self._arrays[metadata["presence"]]
            if means.dtype != np.float32 or means.shape != (count, int(schema["width"])) or not np.isfinite(means).all() or presence.dtype != np.bool_ or presence.shape != (count,) or not presence.any():
                raise ValueError("compact GMM element means or presence are invalid")

    @property
    def frame_p(self) -> np.ndarray:
        return self._arrays["probabilities"]

    @property
    def frame_statistics(self) -> np.ndarray:
        return self._arrays["statistics"]

    @property
    def frame_means(self) -> np.ndarray | None:
        return self._arrays.get("means")

    @property
    def values(self) -> Mapping[str, float]:
        return _OrdinalMapping(self.identities, self.frame_p)

    @property
    def means(self) -> Mapping[str, np.ndarray] | None:
        rows = self.frame_means
        return None if rows is None else _OrdinalMapping(self.identities, rows)

    @property
    def element_means(self):
        if self._elements is None:
            return None
        return {
            element: (
                tuple(self.identities[index] for index in np.flatnonzero(self._arrays[metadata["presence"]])),
                np.asarray(self._arrays[metadata["means"]][self._arrays[metadata["presence"]]], dtype=np.float32),
            ) for element, metadata in self._elements.items()
        }

    def iter_probabilities(self) -> Iterator[tuple[str, float]]:
        return (
            (identity, float(self.frame_p[ordinal]))
            for ordinal, identity in enumerate(self.identities)
        )

    def gather_means(self, identities: Sequence[str]) -> dict[str, np.ndarray]:
        rows = self.frame_means
        if rows is None:
            raise ValueError("compact GMM result has no frame means")
        return {
            identity: np.asarray(rows[self.identities.index(identity)], dtype=np.float32)
            for identity in identities
        }

    def close(self, *, cleanup: bool = False) -> None:
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()
        if cleanup:
            shutil.rmtree(self.root, ignore_errors=True)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["CompactScoreView"]
