from __future__ import annotations

from workflow_backends.sampling.base import SamplingExecutionBackend


_BACKEND_CACHE: dict[str, SamplingExecutionBackend] = {}


def _backend_factories():
    from workflow_backends.sampling.md.gpumd import GPUMDSamplingBackend
    from workflow_backends.sampling.md.lammps import LAMMPSSamplingBackend

    return {
        "gpumd": GPUMDSamplingBackend,
        "lammps": LAMMPSSamplingBackend,
    }


def get_sampling_backend(name: str | None) -> SamplingExecutionBackend:
    normalized = str(name or "").strip().lower()
    factories = _backend_factories()
    backend_cls = factories.get(normalized)
    if backend_cls is None:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unknown sampling_backend: {name!r}. Supported sampling_backend values: {supported}")
    if normalized not in _BACKEND_CACHE:
        backend = backend_cls()
        if not isinstance(backend, SamplingExecutionBackend):
            raise TypeError(f"Sampling backend {name!r} does not implement sampling execution")
        _BACKEND_CACHE[normalized] = backend
    return _BACKEND_CACHE[normalized]
