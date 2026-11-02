"""Pickleable, in-memory decision tasks shared by selection and its replay."""
from __future__ import annotations

from workflow_policies.pair_guard import PairGuardThresholds, evaluate_pair_guard
from workflow_policies.selection import (
    select_element_joint_representatives,
    select_element_representatives,
    select_joint_representatives_with_audit,
)


def pair_guard_task(payload):
    frames, minimum, pairs = payload
    return evaluate_pair_guard(frames, PairGuardThresholds(minimum, pairs))


def joint_selection_task(payload):
    names, vectors, background_names, background, keep = payload
    return select_joint_representatives_with_audit(
        names, vectors, background_names, background, keep=keep,
    )


def element_joint_selection_task(payload):
    names, rows, presence, background_names, background_rows, background_presence, elements, keep = payload
    result = select_element_joint_representatives(
        names, rows, candidate_presence=presence,
        background_identities=background_names,
        background_element_rows=background_rows,
        background_presence=background_presence, type_map=elements, keep=keep,
    )
    return result.selected_identities, dict(result.audit)


def element_selection_task(payload):
    names, rows, presence, elements, keep = payload
    result = select_element_representatives(
        names, rows, presence=presence, type_map=elements, keep=keep,
    )
    return result.selected_identities, dict(result.audit)
