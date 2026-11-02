from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ase import Atoms

from workflow_artifacts.lifecycle.pair_guard import publish_pair_guard
from workflow_core.structures.formats import write_extxyz_structures


class ScreenArtifactPublicationCapability:
    """Schema-specific screen writes scoped to a caller-owned atomic stage."""

    def write_structures(
        self, path: Path, frames: Sequence[Atoms], *, allow_empty: bool
    ) -> Path:
        return write_extxyz_structures(path, frames, allow_empty=allow_empty)

    def write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def publish_pair_guard(self, path: Path, decisions, *, thresholds) -> None:
        publish_pair_guard(path, decisions, thresholds=thresholds)

    def publish_gmm_screen_scores(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.gmm_screen import publish_gmm_screen_scores

        publish_gmm_screen_scores(*args, **kwargs)

    def validate_gmm_pair_guard_alignment(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.gmm_screen import (
            validate_gmm_pair_guard_alignment,
        )

        validate_gmm_pair_guard_alignment(*args, **kwargs)

    def publish_screen_report(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.screen_report import publish_screen_report

        publish_screen_report(*args, **kwargs)

    def publish_frame_mean_representatives(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.representative_features import (
            publish_frame_mean_representatives,
        )

        publish_frame_mean_representatives(*args, **kwargs)

    def publish_joint_direct_screen_evidence(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.joint_direct import (
            publish_joint_direct_screen_evidence,
        )

        publish_joint_direct_screen_evidence(*args, **kwargs)

    def publish_element_direct_screen_evidence(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.element_direct import (
            publish_element_direct_screen_evidence,
        )

        publish_element_direct_screen_evidence(*args, **kwargs)

    def publish_student_first_single_teacher_committee_student_audit(
        self, *args, **kwargs
    ) -> None:
        from workflow_artifacts.lifecycle.mixed_route import (
            publish_student_first_single_teacher_committee_student_audit,
        )

        publish_student_first_single_teacher_committee_student_audit(
            *args, **kwargs
        )

    def publish_committee_teacher_single_student_audit(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.mixed_route import (
            publish_committee_teacher_single_student_audit,
        )

        publish_committee_teacher_single_student_audit(*args, **kwargs)

    def mixed_route_summary(self, *args, **kwargs):
        from workflow_artifacts.lifecycle.mixed_route import mixed_route_summary

        return mixed_route_summary(*args, **kwargs)

    def committee_teacher_single_student_summary(self, *args, **kwargs):
        from workflow_artifacts.lifecycle.mixed_route import (
            committee_teacher_single_student_summary,
        )

        return committee_teacher_single_student_summary(*args, **kwargs)

    def publish_committee_force_audit(self, *args, **kwargs) -> None:
        from workflow_artifacts.lifecycle.committee_force import (
            publish_committee_force_audit,
        )

        publish_committee_force_audit(*args, **kwargs)

    def committee_force_summary(self, *args, **kwargs):
        from workflow_artifacts.lifecycle.committee_force import committee_force_summary

        return committee_force_summary(*args, **kwargs)

    @property
    def gmm_summary_identity(self) -> tuple[str, int]:
        from workflow_artifacts.lifecycle.gmm_screen import (
            SUMMARY_SCHEMA,
            SUMMARY_SCHEMA_REVISION,
        )

        return SUMMARY_SCHEMA, SUMMARY_SCHEMA_REVISION

    def router_provenance(self, router: Path | None) -> dict[str, object]:
        if router is None:
            return {
                "model_digest": None,
                "reference_digest": None,
                "descriptor_schema": None,
            }
        manifest = json.loads(
            (Path(router) / "manifest.json").read_text(encoding="utf-8")
        )
        return {
            "model_digest": manifest["model_digest"],
            "reference_digest": manifest["reference_digest"],
            "descriptor_schema": manifest["descriptor_schema"],
        }


__all__ = ["ScreenArtifactPublicationCapability"]
