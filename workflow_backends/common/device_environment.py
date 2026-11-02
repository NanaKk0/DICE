"""Environment and process ownership for a single assigned GPU session."""
from dataclasses import dataclass
from typing import Mapping
from workflow_backends.execution import ProcessGroupRegistry

@dataclass(frozen=True, slots=True)
class DeviceSessionEnvironment(Mapping[str, str]):
    """Bind one physical assignment to a device-session-local client ordinal."""

    assigned_visible_device: str
    client_visible_device: str
    overrides: Mapping[str, str]
    process_registry: ProcessGroupRegistry

    def __post_init__(self) -> None:
        if not self.assigned_visible_device or not self.client_visible_device:
            raise ValueError("device session visibility identities must be non-empty")
        if "CUDA_VISIBLE_DEVICES" in self.overrides:
            raise ValueError(
                "device session overrides must not duplicate client visibility"
            )

    def __getitem__(self, key: str) -> str:
        if key == "CUDA_VISIBLE_DEVICES":
            return self.client_visible_device
        return self.overrides[key]

    def __iter__(self):
        return iter((*self.overrides, "CUDA_VISIBLE_DEVICES"))

    def __len__(self) -> int:
        return len(self.overrides) + 1

    def cancel_active(self) -> None:
        self.process_registry.cancel_all()

