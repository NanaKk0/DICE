from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def test_structure_id_generation_uses_public_sha256_digest() -> None:
    from workflow_core.structures.canonical_identity import structure_id_for_atoms

    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    shifted_below_rounding = atoms.copy()
    shifted_below_rounding.positions[1, 0] += 1e-10

    structure_id = structure_id_for_atoms(atoms)

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", structure_id)
    assert structure_id_for_atoms(shifted_below_rounding) == structure_id


def test_formula_system_id_and_structure_source_validation() -> None:
    from workflow_core.structures.canonical_identity import (
        formula_system_id,
        validate_structure_source,
    )

    assert formula_system_id(Atoms("OH2")) == "formula:H2O"
    assert validate_structure_source("external") == "external"
    assert validate_structure_source("workflow") == "workflow"

    try:
        validate_structure_source("relabel")
    except ValueError as exc:
        assert "structure_source" in str(exc)
        assert "external" in str(exc)
        assert "workflow" in str(exc)
    else:
        raise AssertionError("invalid structure_source should fail")


def test_stamp_and_validate_required_canonical_identity() -> None:
    from workflow_core.structures.canonical_identity import (
        stamp_canonical_identity,
        validate_canonical_identity,
    )

    atoms = Atoms("Pt", positions=[[0.0, 0.0, 0.0]])
    stamped = stamp_canonical_identity(atoms, system_id="pt111", structure_source="external")

    assert stamped is not atoms
    assert stamped.info["system_id"] == "pt111"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", stamped.info["structure_id"])
    assert stamped.info["structure_source"] == "external"
    assert validate_canonical_identity(stamped) == {
        "system_id": "pt111",
        "structure_id": stamped.info["structure_id"],
        "structure_source": "external",
    }

    missing_source = stamped.copy()
    del missing_source.info["structure_source"]
    try:
        validate_canonical_identity(missing_source)
    except ValueError as exc:
        assert "structure_source" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing required canonical identity should fail")

    malformed_id = stamped.copy()
    malformed_id.info["structure_id"] = "seed-a"
    try:
        validate_canonical_identity(malformed_id)
    except ValueError as exc:
        assert "structure_id" in str(exc)
        assert "sha256:<64hex>" in str(exc)
    else:
        raise AssertionError("malformed structure_id should fail")


def test_identity_audit_payload_represents_contract_events() -> None:
    from workflow_core.structures.canonical_identity import IdentityAudit

    audit = IdentityAudit()
    audit.record_generated_structure_id("sha256:" + "a" * 64)
    audit.record_formula_fallback(system_id="formula:H2O", formula="H2O")
    audit.record_declared_system_overwrite(
        field="seed_system_id",
        previous="legacy_water",
        replacement="declared_water",
    )
    audit.record_ignored_external(route="teacher_dft_candidate_filtering", count=3)
    audit.record_external_dedup_drop(route="candidate", count=2)

    assert audit.to_payload() == {
        "generated_structure_ids": [{"structure_id": "sha256:" + "a" * 64}],
        "formula_fallbacks": [{"system_id": "formula:H2O", "formula": "H2O"}],
        "declared_system_overwrites": [
            {"field": "seed_system_id", "previous": "legacy_water", "replacement": "declared_water"}
        ],
        "ignored_external_counts": {"teacher_dft_candidate_filtering": 3},
        "external_dedup_dropped_counts": {"candidate": 2},
    }


def test_external_ingress_preserves_explicit_systems_and_audits_formula_fallbacks() -> None:
    from workflow_core.structures.canonical_identity import IdentityAudit, stamp_external_ingress_identity

    explicit = Atoms("Pt", positions=[[0.0, 0.0, 0.0]])
    explicit.info["system_id"] = "declared_pt"
    missing = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    audit = IdentityAudit()

    explicit_stamped = stamp_external_ingress_identity(explicit, audit=audit)
    fallback_stamped = stamp_external_ingress_identity(missing, audit=audit)

    assert explicit_stamped.info["system_id"] == "declared_pt"
    assert explicit_stamped.info["structure_source"] == "external"
    assert fallback_stamped.info["system_id"] == "formula:H2O"
    assert fallback_stamped.info["structure_source"] == "external"
    assert audit.to_payload()["formula_fallbacks"] == [{"system_id": "formula:H2O", "formula": "H2O"}]


def test_external_ingress_preserves_singlepoint_labels_when_written(tmp_path: Path) -> None:
    from workflow_core.structures.canonical_identity import IdentityAudit, stamp_external_ingress_identity
    from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures

    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.2], [0.0, -0.2, -0.5]])
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    atoms.calc = SinglePointCalculator(atoms, energy=-76.4, forces=forces, stress=stress)

    stamped = stamp_external_ingress_identity(atoms, audit=IdentityAudit())
    output = write_extxyz_structures(tmp_path / "stamped.xyz", [stamped], allow_empty=False)
    [roundtrip] = read_extxyz_structures(output, allow_empty=False)

    assert roundtrip.calc is not None
    assert roundtrip.get_potential_energy() == -76.4
    np.testing.assert_allclose(roundtrip.get_forces(), forces)
    np.testing.assert_allclose(roundtrip.calc.results["stress"], stress)


def test_canonical_annotation_does_not_write_retired_seed_aliases() -> None:
    from workflow_core.structures.formats import annotate_canonical_structure

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["system_id"] = "water"
    atoms.calc = SinglePointCalculator(atoms, energy=-0.5, forces=np.zeros((1, 3)))

    annotated = annotate_canonical_structure(atoms, force_std_max=0.4, route="candidate")

    assert annotated.info["system_id"] == "water"
    assert "structure_id" in annotated.info
    assert annotated.info["route"] == "candidate"
    assert annotated.info["force_std_max"] == 0.4
    assert "seed_system_id" not in annotated.info
    assert "seed_structure_id" not in annotated.info
    assert annotated.get_potential_energy() == -0.5
    np.testing.assert_allclose(annotated.get_forces(), np.zeros((1, 3)))


def test_active_canonical_paths_do_not_call_legacy_seed_metadata_helpers() -> None:
    roots = (
        Path("workflow_core/v2"),
        Path("workflow_core/structures"),
        Path("workflow_policies/v2"),
        Path("workflow_artifacts/v2"),
    )
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "ensure_minimal_seed_metadata" not in source, str(path)


def test_active_canonical_extxyz_paths_do_not_assign_retired_alias_fields() -> None:
    roots = (
        Path("workflow_core/v2"),
        Path("workflow_core/structures"),
        Path("workflow_policies/v2"),
        Path("workflow_artifacts/v2"),
    )
    assignment_tokens = [
        'info["seed_system_id"] =',
        "info['seed_system_id'] =",
        'info["seed_structure_id"] =',
        "info['seed_structure_id'] =",
        'info["source_system_id"] =',
        "info['source_system_id'] =",
        'info["source_structure_id"] =',
        "info['source_structure_id'] =",
    ]

    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in assignment_tokens:
                assert token not in source, str(path)


# The upstream documentation-maintenance assertion is not part of this
# distribution's runtime test subset; internal agent contracts are not shipped.
