from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_backends.deepmd_common.commands import build_dp_train_command


@dataclass(frozen=True, slots=True)
class DeepMDModelFamilyAdapter:
    name: str
    model_suffix: str

    def primary_model_name(self, teacher_index: int) -> str:
        return f"deepmd_teacher_{teacher_index:02d}{self.model_suffix}"

    def foundation_model_name(self) -> str:
        return f"deepmd_foundation{self.model_suffix}"

    def default_train_command(self) -> list[str]:
        return ["dp", "train"]

    def default_freeze_command(self) -> list[str]:
        return ["dp", "freeze", "-o", "frozen_model"]

    def train_command(self, dp_executable: str) -> list[str]:
        return [str(dp_executable), "train"]

    def freeze_command(self, dp_executable: str) -> list[str]:
        return [str(dp_executable), "freeze", "-o", "frozen_model"]

    def build_train_command(
        self,
        train_command: list[str],
        input_json_name: str,
        *,
        init_mode: str,
        foundation_model: str | Path | None = None,
    ) -> list[str]:
        del init_mode
        return build_dp_train_command(train_command, input_json_name, finetune_model=foundation_model)

    def freeze_output_model_name(self) -> str:
        return f"frozen_model{self.model_suffix}"

    def validate_template(self, template: dict[str, Any], *, type_map: list[str], template_path: Path) -> None:
        del template, type_map, template_path

    def render_member_input(
        self,
        template: dict[str, Any],
        *,
        type_map: list[str],
        train_systems: list[str],
        valid_systems: list[str],
        seed: int,
    ) -> dict[str, Any]:
        payload = copy.deepcopy(template)
        payload.setdefault("model", {})["type_map"] = list(type_map)
        training = payload.setdefault("training", {})
        training.setdefault("training_data", {})["systems"] = list(train_systems)
        training.setdefault("validation_data", {})["systems"] = list(valid_systems)
        training["seed"] = int(seed) + 2000
        return payload


class DPA4FamilyAdapter(DeepMDModelFamilyAdapter):
    def __init__(self) -> None:
        super().__init__(name="dpa4", model_suffix=".pt2")

    def default_train_command(self) -> list[str]:
        return ["dp", "--pt", "train"]

    def default_freeze_command(self) -> list[str]:
        return ["dp", "--pt", "freeze", "-c", "model.ckpt.pt", "-o", "frozen_model"]

    def train_command(self, dp_executable: str) -> list[str]:
        return [str(dp_executable), "--pt", "train"]

    def freeze_command(self, dp_executable: str) -> list[str]:
        return [str(dp_executable), "--pt", "freeze", "-c", "model.ckpt.pt", "-o", "frozen_model"]

    def build_train_command(
        self,
        train_command: list[str],
        input_json_name: str,
        *,
        init_mode: str,
        foundation_model: str | Path | None = None,
    ) -> list[str]:
        if init_mode == "scratch":
            return build_dp_train_command(train_command, input_json_name)
        if init_mode == "foundation_finetune":
            if foundation_model is None:
                raise ValueError("DPA4 foundation_finetune requires a foundation_model")
            return build_dp_train_command(train_command, input_json_name, finetune_model=foundation_model)
        raise ValueError(f"Unsupported DPA4 teacher init_mode={init_mode!r}")

    def validate_template(self, template: dict[str, Any], *, type_map: list[str], template_path: Path) -> None:
        model = template.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"DPA4 model_template missing model object: {template_path}")
        descriptor = model.get("descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError(f"DPA4 model_template missing model.descriptor object: {template_path}")
        if str(model.get("type", "")).strip().lower() != "dpa4":
            raise ValueError(f"DPA4 model_template requires model.type='dpa4': {template_path}")
        if str(descriptor.get("type", "")).strip().lower() != "dpa4":
            raise ValueError(f"DPA4 model_template requires model.descriptor.type='dpa4': {template_path}")

        template_type_map = model.get("type_map", [])
        if template_type_map not in (None, []):
            normalized = [str(item) for item in template_type_map]
            if normalized != list(type_map):
                raise ValueError(
                    f"DPA4 model_template model.type_map must be empty or match DeePMD teacher type_map: {template_path}"
                )

        training = template.get("training")
        if not isinstance(training, dict):
            raise ValueError(f"DPA4 model_template missing training object: {template_path}")
        for key in ("training_data", "validation_data"):
            data_block = training.get(key)
            if not isinstance(data_block, dict):
                raise ValueError(f"DPA4 model_template missing training.{key} object: {template_path}")
            systems = data_block.get("systems", [])
            if systems not in (None, []):
                raise ValueError(
                    f"DPA4 model_template must not own concrete training.{key}.systems: {template_path}"
                )

    def render_member_input(
        self,
        template: dict[str, Any],
        *,
        type_map: list[str],
        train_systems: list[str],
        valid_systems: list[str],
        seed: int,
    ) -> dict[str, Any]:
        payload = super().render_member_input(
            template,
            type_map=type_map,
            train_systems=train_systems,
            valid_systems=valid_systems,
            seed=seed,
        )
        model = payload.setdefault("model", {})
        descriptor = model.setdefault("descriptor", {})
        fitting_net = model.setdefault("fitting_net", {})
        descriptor["seed"] = int(seed)
        fitting_net["seed"] = int(seed) + 1000
        return payload


_ADAPTERS = {
    "dpa4": DPA4FamilyAdapter(),
}


def get_deepmd_family_adapter(model_family: str) -> DeepMDModelFamilyAdapter:
    normalized = str(model_family or "").strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unsupported DeePMD teacher model_family {model_family!r}; supported: {supported}") from exc


def load_and_validate_model_template(template_path: str | Path, *, adapter: DeepMDModelFamilyAdapter, type_map: list[str]) -> dict[str, Any]:
    resolved = Path(template_path).resolve()
    template = read_json(resolved)
    adapter.validate_template(template, type_map=type_map, template_path=resolved)
    return template


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
