from __future__ import annotations

from workflow_core.identities import RevisionedIdentity


GMM_REFERENCE_TAIL = RevisionedIdentity("gmm-reference-tail", 4)
PAIR_GUARD = RevisionedIdentity("pair-guard", 1)
TEACHER_DEVIATION_STUDENT_REFERENCE = RevisionedIdentity(
    "teacher-force-deviation_student-reference-tail", 1
)
STUDENT_REFERENCE_TEACHER_REFERENCE = RevisionedIdentity(
    "student-reference-tail_teacher-reference-tail", 1
)
STUDENT_DEVIATION_TEACHER_REFERENCE = RevisionedIdentity(
    "student-force-deviation_teacher-reference-tail", 1
)
READOUT_INPUT_FRAME_MEAN = RevisionedIdentity("readout-input-frame-mean", 1)
VALIDATED_FINAL_LAYER_NAME = RevisionedIdentity("validated-final-layer-name", 1)
TORCHNEP_READOUT_INPUT = RevisionedIdentity("torchnep-readout-input", 1)
JOINT_DIRECT = RevisionedIdentity("joint-direct", 2)
ELEMENT_DIRECT = RevisionedIdentity("element-direct", 1)
ELEMENT_JOINT_DIRECT = RevisionedIdentity("element-joint-direct", 1)

SELECTORS = {
    "DIRECT": RevisionedIdentity("DIRECT", 3),
    "JOINT_DIRECT": JOINT_DIRECT,
    "ELEMENT_DIRECT": ELEMENT_DIRECT,
    "ELEMENT_JOINT_DIRECT": ELEMENT_JOINT_DIRECT,
    "FPS": RevisionedIdentity("FPS", 1),
    "random": RevisionedIdentity("random", 1),
}


def selector_identity(name: str) -> RevisionedIdentity:
    requested = str(name)
    try:
        return SELECTORS[requested]
    except KeyError as exc:
        matches = tuple(
            identity for identity in SELECTORS.values() if identity.name == requested
        )
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"unsupported selector {name!r}") from exc


__all__ = [
    "ELEMENT_DIRECT",
    "ELEMENT_JOINT_DIRECT",
    "GMM_REFERENCE_TAIL",
    "JOINT_DIRECT",
    "PAIR_GUARD",
    "READOUT_INPUT_FRAME_MEAN",
    "SELECTORS",
    "STUDENT_REFERENCE_TEACHER_REFERENCE",
    "STUDENT_DEVIATION_TEACHER_REFERENCE",
    "TEACHER_DEVIATION_STUDENT_REFERENCE",
    "TORCHNEP_READOUT_INPUT",
    "VALIDATED_FINAL_LAYER_NAME",
    "selector_identity",
]
