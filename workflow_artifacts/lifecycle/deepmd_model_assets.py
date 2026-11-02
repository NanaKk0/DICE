"""Strict role-aware receipts for ordinary DeepMD-PT/DPA4 members.

The canonical ``.pt2`` remains the public inference terminal.  Ordinary DPA4
readout extraction additionally needs the exact checkpoint passed to freeze;
this module owns that relationship without teaching generic model handles
about backend-specific dual representations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


DEEPMD_MODEL_ASSET_SCHEMA = "dice.lifecycle.deepmd-model-assets"
DEEPMD_MODEL_ASSET_SCHEMA_REVISION = 1
DPA4_MODEL_FAMILY = "dpa4"
DPA4_TERMINAL_FREEZE_RELATIONSHIP = "terminal_frozen_from_checkpoint"
DPA4_TERMINAL_NAME = "teacher_model.pt2"
DPA4_READOUT_SOURCE_NAME = "teacher_readout_checkpoint.pt"

_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "schema_revision",
        "member_id",
        "member_order",
        "model_family",
        "native_type_map",
        "inference_terminal_path",
        "inference_terminal_sha256",
        "readout_source_path",
        "readout_source_sha256",
        "checkpoint_family",
        "selected_checkpoint_argument",
        "relationship",
        "freeze_command_profile",
    }
)


@dataclass(frozen=True, slots=True)
class DeepMDModelRoleView:
    """Validated role paths for one canonical DeepMD member."""

    member_id: str
    member_order: int
    model_family: str
    native_type_map: tuple[str, ...]
    inference_terminal: Path
    readout_source: Path
    checkpoint_family: str
    selected_checkpoint_argument: str
    relationship: str
    freeze_command_profile: Mapping[str, Any]
    manifest_path: Path
    inference_terminal_sha256: str
    readout_source_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        member_id = _validated_member_id(self.member_id)
        if isinstance(self.member_order, bool) or not isinstance(self.member_order, int) or self.member_order <= 0:
            raise ValueError("DeepMD model asset member_order must be a positive integer")
        if self.model_family != DPA4_MODEL_FAMILY:
            raise ValueError("DeepMD split model assets are supported only for ordinary DPA4")
        native_type_map = _validated_type_map(self.native_type_map)
        if self.checkpoint_family not in {"regular", "ema"}:
            raise ValueError("DeepMD DPA4 checkpoint family must be regular or ema")
        if not self.selected_checkpoint_argument.strip():
            raise ValueError("DeepMD DPA4 selected checkpoint argument must be non-empty")
        if self.relationship != DPA4_TERMINAL_FREEZE_RELATIONSHIP:
            raise ValueError("DeepMD DPA4 model asset relationship is unsupported")
        profile = _validated_profile(self.freeze_command_profile)
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "native_type_map", native_type_map)
        object.__setattr__(self, "inference_terminal", Path(self.inference_terminal).resolve())
        object.__setattr__(self, "readout_source", Path(self.readout_source).resolve())
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        object.__setattr__(self, "freeze_command_profile", MappingProxyType(profile))

    @property
    def inference_terminal_path(self) -> Path:
        return self.inference_terminal

    @property
    def readout_source_path(self) -> Path:
        return self.readout_source


@dataclass(frozen=True, slots=True)
class DeepMDModelBundleInput:
    """Typed staged input consumed when the lifecycle seals a member receipt."""

    model_family: str
    native_type_map: tuple[str, ...]
    readout_source_path: Path
    checkpoint_family: str
    selected_checkpoint_argument: str
    freeze_command_profile: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model_family != DPA4_MODEL_FAMILY:
            raise ValueError("DeepMD split bundle input is supported only for ordinary DPA4")
        object.__setattr__(self, "native_type_map", _validated_type_map(self.native_type_map))
        source = Path(self.readout_source_path)
        if source.is_symlink() or source.suffix != ".pt" or not source.is_file() or source.stat().st_size <= 0:
            raise ValueError("DeepMD DPA4 readout source must be a non-empty regular .pt file")
        object.__setattr__(self, "readout_source_path", source.resolve())
        if self.checkpoint_family not in {"regular", "ema"}:
            raise ValueError("DeepMD DPA4 checkpoint family must be regular or ema")
        if not isinstance(self.selected_checkpoint_argument, str) or not self.selected_checkpoint_argument.strip():
            raise ValueError("DeepMD DPA4 selected checkpoint argument must be non-empty")
        object.__setattr__(self, "freeze_command_profile", MappingProxyType(_validated_profile(self.freeze_command_profile)))


# A descriptive alias used by backend callers that want to emphasize that the
# receipt is a member asset rather than a generic lifecycle model handle.
DeepMDMemberAsset = DeepMDModelRoleView
DeepMDModelBundleMember = DeepMDModelRoleView


def publish_deepmd_model_bundle(
    *,
    member_root: str | Path,
    member_id: str,
    member_order: int,
    model_family: str,
    native_type_map: Sequence[str],
    inference_terminal_path: str | Path,
    readout_source_path: str | Path,
    checkpoint_family: str,
    selected_checkpoint_argument: str,
    freeze_command_profile: Mapping[str, Any],
) -> DeepMDModelRoleView:
    """Atomically seal one staged ordinary-DPA4 member relationship receipt."""

    root = _validated_member_root(member_root)
    member_id = _validated_member_id(member_id)
    if model_family != DPA4_MODEL_FAMILY:
        raise ValueError(
            "DeepMD model bundle publication is only valid for ordinary DPA4; "
            "DPA4C and other backends remain terminal-only"
        )
    if isinstance(member_order, bool) or not isinstance(member_order, int) or member_order <= 0:
        raise ValueError("DeepMD model asset member_order must be a positive integer")
    type_map = _validated_type_map(native_type_map)
    terminal = _validated_canonical_file(
        inference_terminal_path,
        root=root,
        expected_name=DPA4_TERMINAL_NAME,
        expected_suffix=".pt2",
        role="inference terminal",
    )
    checkpoint = _validated_canonical_file(
        readout_source_path,
        root=root,
        expected_name=DPA4_READOUT_SOURCE_NAME,
        expected_suffix=".pt",
        role="readout source",
    )
    manifest = root / "deepmd_model_assets.json"
    if manifest.exists() or manifest.is_symlink():
        raise ValueError(f"DeepMD model asset receipt already exists: {manifest}")
    payload = {
        "schema": DEEPMD_MODEL_ASSET_SCHEMA,
        "schema_revision": DEEPMD_MODEL_ASSET_SCHEMA_REVISION,
        "member_id": member_id,
        "member_order": member_order,
        "model_family": model_family,
        "native_type_map": list(type_map),
        "inference_terminal_path": terminal.name,
        "inference_terminal_sha256": file_sha256(terminal),
        "readout_source_path": checkpoint.name,
        "readout_source_sha256": file_sha256(checkpoint),
        "checkpoint_family": checkpoint_family,
        "selected_checkpoint_argument": selected_checkpoint_argument,
        "relationship": DPA4_TERMINAL_FREEZE_RELATIONSHIP,
        "freeze_command_profile": _validated_profile(freeze_command_profile),
    }
    _validated_receipt_payload(payload)
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"DeepMD model asset receipt temporary path already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return _load_member_bundle(
        root,
        expected_member_id=member_id,
        expected_member_order=member_order,
        expected_native_type_map=type_map,
    )


def load_deepmd_model_bundle(
    *,
    member_root: str | Path,
    expected_member_id: str | None = None,
    expected_member_order: int | None = None,
    expected_native_type_map: Sequence[str] | None = None,
) -> DeepMDModelRoleView:
    """Load and fully attest one exact member-local DPA4 receipt."""

    root = _validated_member_root(member_root)
    return _load_member_bundle(
        root,
        expected_member_id=expected_member_id,
        expected_member_order=expected_member_order,
        expected_native_type_map=None
        if expected_native_type_map is None
        else _validated_type_map(expected_native_type_map),
    )


def load_deepmd_model_bundles(
    *,
    stage_root: str | Path,
    expected_member_ids: Sequence[str],
    expected_native_type_maps: Sequence[Sequence[str]] | None = None,
) -> tuple[DeepMDModelRoleView, ...]:
    """Resolve exactly the expected ordered members without directory scans."""

    root = Path(stage_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"DeepMD model bundle stage root must be a regular directory: {root}")
    member_ids = tuple(_validated_member_id(value) for value in expected_member_ids)
    if not member_ids or len(set(member_ids)) != len(member_ids):
        raise ValueError("DeepMD model bundle requires ordered unique member identities")
    if expected_native_type_maps is not None:
        maps = tuple(_validated_type_map(value) for value in expected_native_type_maps)
        if len(maps) != len(member_ids):
            raise ValueError("DeepMD model bundle native type-map coverage is incomplete")
    else:
        maps = (None,) * len(member_ids)
    resolved = tuple(
        _load_member_bundle(
            root / member_id,
            expected_member_id=member_id,
            expected_member_order=index,
            expected_native_type_map=maps[index - 1],
        )
        for index, member_id in enumerate(member_ids, start=1)
    )
    if tuple(item.member_id for item in resolved) != member_ids:
        raise ValueError("DeepMD model bundle member coverage/order changed")
    if len({item.inference_terminal for item in resolved}) != len(resolved):
        raise ValueError("DeepMD model bundle terminal paths are not unique")
    if len({item.readout_source for item in resolved}) != len(resolved):
        raise ValueError("DeepMD model bundle readout paths are not unique")
    return resolved


# Resolver-oriented aliases keep the role vocabulary explicit at call sites.
resolve_deepmd_model_bundle = load_deepmd_model_bundle
resolve_deepmd_model_bundles = load_deepmd_model_bundles
load_deepmd_member_assets = load_deepmd_model_bundle
resolve_deepmd_member_assets = load_deepmd_model_bundle


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_member_bundle(
    root: Path,
    *,
    expected_member_id: str | None,
    expected_member_order: int | None,
    expected_native_type_map: Sequence[str] | None,
) -> DeepMDModelRoleView:
    root = _validated_member_root(root)
    manifest = root / "deepmd_model_assets.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size <= 0:
        raise ValueError(f"DeepMD model asset receipt is missing or invalid: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"DeepMD model asset receipt is invalid JSON: {manifest}") from exc
    _validated_receipt_payload(payload)
    if expected_member_id is not None and payload["member_id"] != _validated_member_id(expected_member_id):
        raise ValueError("DeepMD model asset member identity changed")
    if expected_member_order is not None and payload["member_order"] != expected_member_order:
        raise ValueError("DeepMD model asset member order changed")
    if expected_native_type_map is not None and tuple(payload["native_type_map"]) != tuple(expected_native_type_map):
        raise ValueError("DeepMD model asset native type map changed")
    terminal = _validated_declared_file(
        root,
        payload["inference_terminal_path"],
        expected_name=DPA4_TERMINAL_NAME,
        expected_suffix=".pt2",
        expected_digest=payload["inference_terminal_sha256"],
        role="inference terminal",
    )
    checkpoint = _validated_declared_file(
        root,
        payload["readout_source_path"],
        expected_name=DPA4_READOUT_SOURCE_NAME,
        expected_suffix=".pt",
        expected_digest=payload["readout_source_sha256"],
        role="readout source",
    )
    return DeepMDModelRoleView(
        member_id=payload["member_id"],
        member_order=payload["member_order"],
        model_family=payload["model_family"],
        native_type_map=tuple(payload["native_type_map"]),
        inference_terminal=terminal,
        readout_source=checkpoint,
        checkpoint_family=payload["checkpoint_family"],
        selected_checkpoint_argument=payload["selected_checkpoint_argument"],
        relationship=payload["relationship"],
        freeze_command_profile=payload["freeze_command_profile"],
        manifest_path=manifest,
        inference_terminal_sha256=payload["inference_terminal_sha256"],
        readout_source_sha256=payload["readout_source_sha256"],
        manifest_sha256=file_sha256(manifest),
    )


def _validated_receipt_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _RECEIPT_KEYS:
        raise ValueError("DeepMD model asset receipt shape is invalid")
    if (
        payload["schema"] != DEEPMD_MODEL_ASSET_SCHEMA
        or payload["schema_revision"] != DEEPMD_MODEL_ASSET_SCHEMA_REVISION
        or isinstance(payload["schema_revision"], bool)
    ):
        raise ValueError("DeepMD model asset receipt identity is unsupported")
    _validated_member_id(payload["member_id"])
    if isinstance(payload["member_order"], bool) or not isinstance(payload["member_order"], int) or payload["member_order"] <= 0:
        raise ValueError("DeepMD model asset member_order is invalid")
    if payload["model_family"] != DPA4_MODEL_FAMILY:
        raise ValueError("DeepMD model asset family is not ordinary DPA4")
    _validated_type_map(payload["native_type_map"])
    if payload["inference_terminal_path"] != DPA4_TERMINAL_NAME:
        raise ValueError("DeepMD model asset inference terminal path is ambiguous or non-canonical")
    if payload["readout_source_path"] != DPA4_READOUT_SOURCE_NAME:
        raise ValueError("DeepMD model asset readout source path is ambiguous or non-canonical")
    _validated_digest(payload["inference_terminal_sha256"], "inference terminal")
    _validated_digest(payload["readout_source_sha256"], "readout source")
    if payload["checkpoint_family"] not in {"regular", "ema"}:
        raise ValueError("DeepMD model asset checkpoint family is invalid")
    if not isinstance(payload["selected_checkpoint_argument"], str) or not payload["selected_checkpoint_argument"].strip():
        raise ValueError("DeepMD model asset selected checkpoint argument is invalid")
    if payload["relationship"] != DPA4_TERMINAL_FREEZE_RELATIONSHIP:
        raise ValueError("DeepMD model asset freeze relationship is invalid")
    _validated_profile(payload["freeze_command_profile"])
    return dict(payload)


def _validated_member_root(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"DeepMD model asset member root must not be a symlink: {raw}")
    root = raw.resolve()
    if not root.is_dir():
        raise ValueError(f"DeepMD model asset member root must be a directory: {root}")
    return root


def _validated_canonical_file(
    path: str | Path,
    *,
    root: Path,
    expected_name: str,
    expected_suffix: str,
    role: str,
) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"DeepMD model asset {role} must be a regular file, not a symlink")
    resolved = raw.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"DeepMD model asset {role} must remain inside its member root") from exc
    if relative.name != expected_name:
        raise ValueError(f"DeepMD model asset {role} must use canonical name {expected_name}")
    if resolved.suffix != expected_suffix or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"DeepMD model asset {role} must be a non-empty {expected_suffix} regular file")
    return resolved


def _validated_declared_file(
    root: Path,
    value: object,
    *,
    expected_name: str,
    expected_suffix: str,
    expected_digest: str,
    role: str,
) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"DeepMD model asset {role} path must be relative")
    raw = root / value
    if raw.is_symlink():
        raise ValueError(f"DeepMD model asset {role} must be a regular file, not a symlink")
    resolved = raw.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"DeepMD model asset {role} path escapes its member root") from exc
    if Path(value).name != expected_name:
        raise ValueError(f"DeepMD model asset {role} path is ambiguous or non-canonical")
    if not resolved.is_file():
        raise ValueError(f"DeepMD model asset {role} is missing or invalid")
    if resolved.suffix != expected_suffix or resolved.stat().st_size <= 0:
        raise ValueError(f"DeepMD model asset {role} has the wrong suffix or is empty")
    if file_sha256(resolved) != expected_digest:
        raise ValueError(f"DeepMD model asset {role} digest drifted")
    return resolved


def _validated_member_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("DeepMD model asset member_id must be a simple non-empty identity")
    return value.strip()


def _validated_type_map(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value or any(not isinstance(item, str) or not item.strip() for item in value) or len(set(value)) != len(value):
        raise ValueError("DeepMD model asset native type map must contain unique symbols")
    return tuple(str(item) for item in value)


def _validated_digest(value: object, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"DeepMD model asset {role} digest is invalid")
    return value


def _validated_profile(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("DeepMD model asset freeze command profile must be a non-empty object")
    normalized = dict(value)
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("DeepMD model asset freeze command profile must be JSON-compatible") from exc
    return normalized


__all__ = [
    "DPA4_MODEL_FAMILY",
    "DPA4_READOUT_SOURCE_NAME",
    "DPA4_TERMINAL_FREEZE_RELATIONSHIP",
    "DPA4_TERMINAL_NAME",
    "DEEPMD_MODEL_ASSET_SCHEMA",
    "DEEPMD_MODEL_ASSET_SCHEMA_REVISION",
    "DeepMDMemberAsset",
    "DeepMDModelBundleInput",
    "DeepMDModelBundleMember",
    "DeepMDModelRoleView",
    "file_sha256",
    "load_deepmd_member_assets",
    "load_deepmd_model_bundle",
    "load_deepmd_model_bundles",
    "publish_deepmd_model_bundle",
    "resolve_deepmd_member_assets",
    "resolve_deepmd_model_bundle",
    "resolve_deepmd_model_bundles",
]
