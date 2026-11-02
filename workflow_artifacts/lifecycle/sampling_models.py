from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SAMPLING_MODEL_ASSET_SCHEMA = "dice.lifecycle.sampling-model-assets"
SAMPLING_MODEL_ASSET_SCHEMA_REVISION = 1
FIXED_COMPRESSION = {
    "step": 0.01,
    "extrapolate": 5,
    "frequency": -1,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EffectiveSamplingModel:
    member_id: str
    source_path: Path
    source_sha256: str
    effective_path: Path
    effective_sha256: str
    deepmd_recipe_kind: str
    cli: str
    training_script: Path | None = None

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("sampling model member_id must be non-empty")
        if self.cli not in {"tf", "pt"}:
            raise ValueError("sampling model CLI must be tf or pt")
        if self.deepmd_recipe_kind not in {"standard", "dpa4c"}:
            raise ValueError("sampling model DeepMD recipe kind is unsupported")
        source = _regular_native_model(self.source_path, cli=self.cli, role="source")
        effective = _regular_native_model(
            self.effective_path, cli=self.cli, role="effective"
        )
        if source.suffix != effective.suffix:
            raise ValueError("sampling source and effective model suffixes must match")
        if file_sha256(source) != self.source_sha256:
            raise ValueError("sampling source model digest mismatch")
        if file_sha256(effective) != self.effective_sha256:
            raise ValueError("sampling effective model digest mismatch")
        training_script = self.training_script
        if training_script is not None:
            training_script = Path(training_script).resolve()
            if (
                training_script.is_symlink()
                or not training_script.is_file()
                or training_script.stat().st_size <= 0
            ):
                raise ValueError(
                    "sampling model training script must be a non-empty regular file"
                )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "effective_path", effective)
        object.__setattr__(self, "training_script", training_script)


@dataclass(frozen=True, slots=True)
class EffectiveSamplingModelPlan:
    models: tuple[EffectiveSamplingModel, ...]
    compressed: bool
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        models = tuple(self.models)
        member_ids = tuple(model.member_id for model in models)
        if (
            not models
            or any(not isinstance(model, EffectiveSamplingModel) for model in models)
            or len(set(member_ids)) != len(member_ids)
        ):
            raise ValueError(
                "effective sampling model plan requires ordered unique typed members"
            )
        manifest = self.manifest_path
        if self.compressed:
            if manifest is None:
                raise ValueError("compressed sampling model plan requires a manifest")
            manifest = Path(manifest).resolve()
            if (
                manifest.is_symlink()
                or not manifest.is_file()
                or manifest.stat().st_size <= 0
            ):
                raise ValueError(
                    "compressed sampling model plan manifest is missing or invalid"
                )
        elif manifest is not None:
            raise ValueError("uncompressed sampling model plan must not have a manifest")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "manifest_path", manifest)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(model.effective_path for model in self.models)

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(model.member_id for model in self.models)


def operation_profile_evidence(context: object) -> dict[str, Any]:
    role = str(getattr(context, "role", "")).strip()
    profile_id = str(getattr(context, "profile_id", "")).strip()
    source_scripts = tuple(getattr(context, "source_scripts", ()))
    executables = getattr(context, "executables", {})
    runtime = getattr(context, "runtime", {})
    if (
        not role
        or not profile_id
        or not all(isinstance(item, Mapping) for item in source_scripts)
        or not isinstance(executables, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        raise ValueError("sampling model compression operation profile is invalid")
    return {
        "role": role,
        "profile_id": profile_id,
        "source_scripts": [dict(item) for item in source_scripts],
        "executables": dict(executables),
        "runtime": dict(runtime),
    }


def publish_sampling_model_assets(
    *,
    sample_dir: str | Path,
    models: Sequence[EffectiveSamplingModel],
    cli: str,
    operation_profile: Mapping[str, Any],
) -> Path:
    root = Path(sample_dir).resolve()
    iter_root = root.parents[1]
    ordered = tuple(models)
    members: list[dict[str, Any]] = []
    for model in ordered:
        try:
            source_relative = model.source_path.relative_to(iter_root)
            derived_relative = model.effective_path.relative_to(root)
            training_relative = (
                None
                if model.training_script is None
                else model.training_script.relative_to(iter_root)
            )
        except ValueError as exc:
            raise ValueError(
                "sampling model asset paths must remain inside the iter/sample roots"
            ) from exc
        members.append(
            {
                "member_id": model.member_id,
                "source_path": source_relative.as_posix(),
                "source_sha256": model.source_sha256,
                "derived_path": derived_relative.as_posix(),
                "derived_sha256": model.effective_sha256,
                "deepmd_recipe_kind": model.deepmd_recipe_kind,
                "cli": model.cli,
                "native_selector": (
                    "pt_expt" if model.deepmd_recipe_kind == "dpa4c" else model.cli
                ),
                "training_script": (
                    None if training_relative is None else training_relative.as_posix()
                ),
            }
        )
    manifest = root / "sampling_model_assets.json"
    if manifest.exists() or manifest.is_symlink():
        raise ValueError("sampling model asset manifest already exists")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": SAMPLING_MODEL_ASSET_SCHEMA,
                "schema_revision": SAMPLING_MODEL_ASSET_SCHEMA_REVISION,
                "cli": cli,
                "compression": dict(FIXED_COMPRESSION),
                "operation_profile": dict(operation_profile),
                "members": members,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest


def load_sampling_model_assets(
    *,
    sample_dir: str | Path,
    source_models: Sequence[object],
    cli: str,
    expected_operation_profile: Mapping[str, Any] | None = None,
) -> EffectiveSamplingModelPlan:
    """Load and fully attest one committed or staging compression manifest."""

    if cli not in {"tf", "pt"}:
        raise ValueError("sampling model asset CLI must be tf or pt")
    root = Path(sample_dir).resolve()
    iter_root = root.parents[1]
    manifest = root / "sampling_model_assets.json"
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size <= 0
    ):
        raise ValueError("sampling model asset manifest is missing or invalid")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("sampling model asset manifest is invalid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "schema_revision",
        "cli",
        "compression",
        "operation_profile",
        "members",
    }:
        raise ValueError("sampling model asset manifest shape is invalid")
    if (
        payload["schema"] != SAMPLING_MODEL_ASSET_SCHEMA
        or payload["schema_revision"] != SAMPLING_MODEL_ASSET_SCHEMA_REVISION
        or isinstance(payload["schema_revision"], bool)
        or payload["cli"] != cli
        or payload["compression"] != FIXED_COMPRESSION
    ):
        raise ValueError("sampling model asset manifest identity is incompatible")
    operation_profile = _validated_operation_profile(
        payload["operation_profile"]
    )
    if operation_profile["role"] != f"student.deepmd.{cli}.compress":
        raise ValueError("sampling model asset compression profile is incompatible")
    if expected_operation_profile is not None and operation_profile != (
        _validated_operation_profile(expected_operation_profile)
    ):
        raise ValueError("sampling model asset operation profile changed")
    member_payloads = payload["members"]
    expected = tuple(source_models)
    if not isinstance(member_payloads, list) or len(member_payloads) != len(expected):
        raise ValueError("sampling model asset member coverage is incomplete")

    models: list[EffectiveSamplingModel] = []
    member_keys = {
        "member_id",
        "source_path",
        "source_sha256",
        "derived_path",
        "derived_sha256",
        "deepmd_recipe_kind",
        "cli",
        "native_selector",
        "training_script",
    }
    for record, source_model in zip(member_payloads, expected, strict=True):
        if not isinstance(record, Mapping) or set(record) != member_keys:
            raise ValueError("sampling model asset member shape is invalid")
        member_id = str(getattr(source_model, "member_id", "")).strip()
        source_expected = Path(getattr(source_model, "path", "")).resolve()
        if record["member_id"] != member_id or record["cli"] != cli:
            raise ValueError("sampling model asset member order or CLI changed")
        source = _contained_relative_path(
            iter_root, record["source_path"], label="source"
        )
        derived = _contained_relative_path(
            root, record["derived_path"], label="derived"
        )
        if source != source_expected:
            raise ValueError("sampling model asset source path changed")
        recipe_kind = record["deepmd_recipe_kind"]
        retained_kind = _retained_deepmd_recipe_kind(
            source.parent / "input.json", member_id=member_id
        )
        if recipe_kind != retained_kind:
            raise ValueError("sampling model asset recipe kind changed")
        if recipe_kind == "dpa4c" and cli != "pt":
            raise ValueError("DPA4C sampling model asset requires PT")
        selector = "pt_expt" if recipe_kind == "dpa4c" else cli
        if record["native_selector"] != selector:
            raise ValueError("sampling model asset native selector changed")
        training_value = record["training_script"]
        if recipe_kind == "dpa4c":
            if training_value is None:
                raise ValueError("DPA4C sampling model asset lacks training script")
            training_script = _contained_relative_path(
                iter_root, training_value, label="training script"
            )
        else:
            if training_value is not None:
                raise ValueError(
                    "standard sampling model asset unexpectedly has training script"
                )
            training_script = None
        source_digest = _sha256_text(record["source_sha256"], label="source")
        derived_digest = _sha256_text(record["derived_sha256"], label="derived")
        models.append(
            EffectiveSamplingModel(
                member_id=member_id,
                source_path=source,
                source_sha256=source_digest,
                effective_path=derived,
                effective_sha256=derived_digest,
                deepmd_recipe_kind=recipe_kind,
                cli=cli,
                training_script=training_script,
            )
        )
    return EffectiveSamplingModelPlan(
        tuple(models), compressed=True, manifest_path=manifest
    )


def validate_local_sampling_model_assets(
    *, sample_dir: str | Path, member_ids: Sequence[str], cli: str
) -> tuple[dict[str, Any], ...]:
    """Validate compressed assets without consulting current train/environment files."""

    if cli not in {"tf", "pt"}:
        raise ValueError("local sampling model assets require tf or pt CLI")
    root = Path(sample_dir).resolve()
    manifest = root / "sampling_model_assets.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size <= 0:
        raise ValueError("sampling model asset manifest is missing or invalid")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "schema_revision", "cli", "compression", "operation_profile", "members"
    }:
        raise ValueError("sampling model asset manifest shape is invalid")
    if (
        payload["schema"] != SAMPLING_MODEL_ASSET_SCHEMA
        or payload["schema_revision"] != SAMPLING_MODEL_ASSET_SCHEMA_REVISION
        or isinstance(payload["schema_revision"], bool)
        or payload["cli"] != cli
        or payload["compression"] != FIXED_COMPRESSION
    ):
        raise ValueError("sampling model asset manifest identity is incompatible")
    profile = _validated_operation_profile(payload["operation_profile"])
    if profile["role"] != f"student.deepmd.{cli}.compress":
        raise ValueError("sampling model asset compression role changed")
    expected_members = tuple(str(item) for item in member_ids)
    records = payload["members"]
    if not isinstance(records, list) or len(records) != len(expected_members):
        raise ValueError("sampling model asset member coverage is incomplete")
    keys = {
        "member_id", "source_path", "source_sha256", "derived_path",
        "derived_sha256", "deepmd_recipe_kind", "cli", "native_selector",
        "training_script",
    }
    normalized: list[dict[str, Any]] = []
    for member_id, record in zip(expected_members, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != keys:
            raise ValueError("sampling model asset member shape is invalid")
        if record["member_id"] != member_id or record["cli"] != cli:
            raise ValueError("sampling model asset member order or CLI changed")
        _sha256_text(record["source_sha256"], label="source")
        derived_digest = _sha256_text(record["derived_sha256"], label="derived")
        source_value = record["source_path"]
        if (
            not isinstance(source_value, str)
            or not source_value
            or Path(source_value).is_absolute()
            or ".." in Path(source_value).parts
        ):
            raise ValueError("sampling model asset source path is invalid")
        derived = _contained_relative_path(root, record["derived_path"], label="derived")
        kind = record["deepmd_recipe_kind"]
        selector = "pt_expt" if kind == "dpa4c" else cli
        if kind not in {"standard", "dpa4c"} or record["native_selector"] != selector:
            raise ValueError("sampling model asset recipe/selector identity changed")
        if kind == "dpa4c" and cli != "pt":
            raise ValueError("DPA4C sampling model asset requires PT")
        training = record["training_script"]
        if (kind == "dpa4c") != (training is not None):
            raise ValueError("sampling model training-script mode changed")
        if training is not None and (
            not isinstance(training, str)
            or not training
            or Path(training).is_absolute()
            or ".." in Path(training).parts
        ):
            raise ValueError("sampling model training-script path is invalid")
        effective = _regular_native_model(derived, cli=cli, role="derived")
        if file_sha256(effective) != derived_digest:
            raise ValueError("sampling derived model digest mismatch")
        normalized.append(
            {"member_id": member_id, "derived_path": effective, "derived_sha256": derived_digest}
        )
    return tuple(normalized)


def _contained_relative_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"sampling model asset {label} path must be relative")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"sampling model asset {label} path escapes its root"
        ) from exc
    return resolved


def _sha256_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"sampling model asset {label} digest is invalid")
    return value


def _validated_operation_profile(value: object) -> dict[str, Any]:
    keys = {
        "role",
        "profile_id",
        "source_scripts",
        "executables",
        "runtime",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("sampling model asset operation profile shape is invalid")
    role = value["role"]
    profile_id = value["profile_id"]
    scripts = value["source_scripts"]
    executables = value["executables"]
    runtime = value["runtime"]
    if (
        not isinstance(role, str)
        or not role.strip()
        or not isinstance(profile_id, str)
        or not profile_id.strip()
        or not isinstance(scripts, list)
        or not isinstance(executables, Mapping)
        or "dp" not in executables
        or not isinstance(runtime, Mapping)
    ):
        raise ValueError("sampling model asset operation profile is invalid")
    normalized_scripts: list[dict[str, str]] = []
    for script in scripts:
        if (
            not isinstance(script, Mapping)
            or set(script) != {"path", "digest"}
            or not isinstance(script["path"], str)
            or not script["path"]
        ):
            raise ValueError(
                "sampling model asset operation profile source script is invalid"
            )
        normalized_scripts.append(
            {
                "path": script["path"],
                "digest": _sha256_text(
                    script["digest"], label="operation source script"
                ),
            }
        )
    normalized_executables = dict(executables)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item
        for key, item in normalized_executables.items()
    ):
        raise ValueError(
            "sampling model asset operation profile executable is invalid"
        )
    return {
        "role": role,
        "profile_id": profile_id,
        "source_scripts": normalized_scripts,
        "executables": normalized_executables,
        "runtime": dict(runtime),
    }


def _retained_deepmd_recipe_kind(path: Path, *, member_id: str) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(
            f"sampling model asset lacks retained input for {member_id}"
        )
    try:
        rendered = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"sampling model asset retained input is invalid for {member_id}"
        ) from exc
    model = rendered.get("model", {}) if isinstance(rendered, Mapping) else {}
    descriptor = model.get("descriptor", {}) if isinstance(model, Mapping) else {}
    return (
        "dpa4c"
        if isinstance(descriptor, Mapping)
        and descriptor.get("type") == "dpa4c"
        else "standard"
    )


def _regular_native_model(path: str | Path, *, cli: str, role: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"sampling {role} model must be a regular file")
    resolved = raw.resolve()
    expected_suffix = ".pb" if cli == "tf" else ".pt2"
    if (
        resolved.suffix != expected_suffix
        or not resolved.is_file()
        or resolved.stat().st_size <= 0
    ):
        raise ValueError(
            f"sampling {role} model must be a non-empty regular {expected_suffix} file"
        )
    return resolved
