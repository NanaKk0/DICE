from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms


@dataclass(frozen=True, slots=True)
class SOAPDescriptorGenerator:
    species: tuple[str, ...]
    r_cut: float
    n_max: int
    l_max: int
    sigma: float
    periodic: bool = True
    sparse: bool = False
    dtype: str = "float32"
    _soap_cache: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("SOAPDescriptorGenerator requires a non-empty species list")

    @property
    def soap_dim(self) -> int:
        soap = self._soap_backend()
        return int(soap.get_number_of_features())

    def create_atomic_descriptor(self, atoms: Atoms) -> np.ndarray:
        if len(atoms) == 0:
            return np.zeros((0, self.soap_dim), dtype=self._dtype())
        soap = self._soap_backend()
        try:
            values = soap.create(atoms, centers=list(range(len(atoms))))
        except TypeError:
            values = soap.create(atoms, positions=list(range(len(atoms))))
        array = np.asarray(values, dtype=self._dtype())
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(atoms):
            raise RuntimeError(f"SOAP atomic descriptor has unexpected shape {array.shape}, expected ({len(atoms)}, soap_dim)")
        return array

    def create_system_descriptor(self, frames: Sequence[Atoms]) -> np.ndarray:
        if not frames:
            return np.zeros((0, 0, self.soap_dim), dtype=self._dtype())
        natoms = len(frames[0])
        for frame in frames:
            if len(frame) != natoms:
                raise ValueError(
                    "SOAPDescriptorGenerator requires fixed natoms within one system; "
                    "split the system or enable a ragged loader"
                )
        descriptors = [self.create_atomic_descriptor(frame) for frame in frames]
        return np.asarray(descriptors, dtype=self._dtype())

    @lru_cache(maxsize=8)
    def _soap_backend(self):  # type: ignore[no-untyped-def]
        if self._soap_cache is not None:
            return self._soap_cache
        try:
            from dscribe.descriptors import SOAP
        except Exception as exc:
            raise RuntimeError("SOAP backend requires dscribe. Install with: pip install dscribe") from exc
        kwargs = {
            "species": list(self.species),
            "r_cut": float(self.r_cut),
            "n_max": int(self.n_max),
            "l_max": int(self.l_max),
            "sigma": float(self.sigma),
            "periodic": bool(self.periodic),
            "sparse": bool(self.sparse),
            "dtype": self.dtype,
        }
        try:
            soap = SOAP(**kwargs)
        except TypeError:
            kwargs["rcut"] = kwargs.pop("r_cut")
            kwargs["nmax"] = kwargs.pop("n_max")
            kwargs["lmax"] = kwargs.pop("l_max")
            soap = SOAP(**kwargs)
        object.__setattr__(self, "_soap_cache", soap)
        return soap

    def _dtype(self) -> np.dtype:
        return np.dtype(self.dtype)


@dataclass(frozen=True, slots=True)
class SoapEncoderConfig:
    species: tuple[str, ...]
    r_cut: float
    n_max: int
    l_max: int
    sigma: float
    periodic: bool = True
    sparse: bool = False
    dtype: str = "float32"


def build_soap_structure_encoder(payload: dict[str, Any] | SoapEncoderConfig) -> Callable[[Sequence[Atoms]], np.ndarray]:
    """Build an Atoms-list encoder that mean-pools atomic SOAP into structure features."""
    cfg = normalize_soap_encoder_config(payload)
    soap = SOAPDescriptorGenerator(
        species=cfg.species,
        r_cut=cfg.r_cut,
        n_max=cfg.n_max,
        l_max=cfg.l_max,
        sigma=cfg.sigma,
        periodic=cfg.periodic,
        sparse=cfg.sparse,
        dtype=cfg.dtype,
    )

    def encode(structures: Sequence[Atoms]) -> np.ndarray:
        if not structures:
            return np.zeros((0, soap.soap_dim), dtype=np.dtype(cfg.dtype))
        return np.asarray([soap.create_atomic_descriptor(item).mean(axis=0) for item in structures], dtype=np.dtype(cfg.dtype))

    return encode


def normalize_soap_encoder_config(payload: dict[str, Any] | SoapEncoderConfig) -> SoapEncoderConfig:
    if isinstance(payload, SoapEncoderConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"SOAP encoder config must be a dict or SoapEncoderConfig, got {type(payload)!r}")
    species = tuple(str(item) for item in payload.get("soap_species", payload.get("species", ())))
    if not species:
        raise ValueError("SOAP encoder requires a non-empty species list")
    return SoapEncoderConfig(
        species=species,
        r_cut=float(payload.get("soap_r_cut", payload.get("r_cut", 5.0))),
        n_max=int(payload.get("soap_n_max", payload.get("n_max", 4))),
        l_max=int(payload.get("soap_l_max", payload.get("l_max", 4))),
        sigma=float(payload.get("soap_sigma", payload.get("sigma", 0.5))),
        periodic=bool(payload.get("soap_periodic", payload.get("periodic", True))),
        sparse=bool(payload.get("soap_sparse", payload.get("sparse", False))),
        dtype=str(payload.get("soap_dtype", payload.get("dtype", "float32"))),
    )
