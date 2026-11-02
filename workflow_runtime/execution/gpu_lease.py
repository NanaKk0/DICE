from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


@dataclass(slots=True)
class LocalGPULease:
    owner: str
    visible_devices: tuple[str, ...]
    lock_paths: tuple[Path, ...]
    _released: bool = False

    @property
    def environment_override(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(self.visible_devices)}

    def release(self) -> None:
        if self._released:
            return
        for path, device in zip(self.lock_paths, self.visible_devices):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("GPU lease evidence changed before release") from exc
            if value != {"owner": self.owner, "device": device}:
                raise RuntimeError("GPU lease ownership changed before release")
        for path in self.lock_paths:
            path.unlink()
        self._released = True

    def __enter__(self) -> LocalGPULease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class LocalGPULeaseManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def acquire(self, *, owner: str, count: int) -> LocalGPULease:
        if not isinstance(owner, str) or not _OWNER.fullmatch(owner):
            raise ValueError("GPU lease owner is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("GPU lease count must be a positive integer")
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None or not raw.strip():
            raise RuntimeError(
                "GPU execution requires scheduler/controller-provided CUDA_VISIBLE_DEVICES"
            )
        devices = tuple(value.strip() for value in raw.split(",") if value.strip())
        if not devices or len(set(devices)) != len(devices):
            raise RuntimeError("CUDA_VISIBLE_DEVICES is empty or ambiguous")
        self.root.mkdir(parents=True, exist_ok=True)
        acquired: list[tuple[str, Path]] = []
        for device in devices:
            path = self.root / f"gpu-{hashlib.sha256(device.encode()).hexdigest()}.lease"
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump({"owner": owner, "device": device}, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            acquired.append((device, path))
            if len(acquired) == count:
                break
        if len(acquired) != count:
            for _, path in acquired:
                path.unlink(missing_ok=True)
            raise RuntimeError("no exclusive GPU allocation is available for this DICE unit")
        return LocalGPULease(
            owner=owner,
            visible_devices=tuple(device for device, _ in acquired),
            lock_paths=tuple(path for _, path in acquired),
        )
