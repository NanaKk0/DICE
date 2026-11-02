"""Closed requested-to-effective selector decisions from known model shape."""

from __future__ import annotations

from dataclasses import dataclass


COMMITTEE_DESCRIPTOR_UNAVAILABLE = "committee_descriptor_unavailable"


@dataclass(frozen=True, slots=True)
class PolicyResolution:
    requested_policy: str
    effective_policy: str
    fallback_reason: str | None

    @property
    def fell_back(self) -> bool:
        return self.fallback_reason is not None


def resolve_element_direct_policy(
    requested_policy: str, *, descriptor_model_shape: str
) -> PolicyResolution:
    """Resolve the sole temporary committee fallback without inspecting evidence."""

    shape = str(descriptor_model_shape)
    if shape not in {"single", "committee"}:
        raise ValueError("descriptor model shape must be single or committee")
    requested = str(requested_policy)
    if requested == "ELEMENT_DIRECT" and shape == "committee":
        return PolicyResolution(
            requested,
            "random",
            COMMITTEE_DESCRIPTOR_UNAVAILABLE,
        )
    return PolicyResolution(requested, requested, None)


__all__ = [
    "COMMITTEE_DESCRIPTOR_UNAVAILABLE",
    "PolicyResolution",
    "resolve_element_direct_policy",
]
