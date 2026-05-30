from __future__ import annotations

__all__: list[str] = []
from workflow_backends.sampling.base import MDSamplingRequest, SamplingExecutionBackend
from workflow_backends.sampling.request_adapter import build_md_sampling_request


def get_sampling_backend(name: str | None):
    from workflow_backends.sampling.registry import get_sampling_backend as _get_sampling_backend

    return _get_sampling_backend(name)


__all__ = ["MDSamplingRequest", "SamplingExecutionBackend", "build_md_sampling_request", "get_sampling_backend"]
