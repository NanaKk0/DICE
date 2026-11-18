"""Real saved acquisition -> ordinary screen, with explicit score/label fakes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_saved_gpu_stage_candidates_reach_screen(tmp_path):
    root = os.environ.get('DICE_GCMC_SAMPLE')
    if not root:
        pytest.skip('explicit completed real GPU sample required')
    assert_saved_candidates_reach_screen(root, tmp_path)


def assert_saved_candidates_reach_screen(root, tmp_path, *, keep=2):
    from workflow_backends.gcmc.adapter import GcmcObservationSource, validate_task
    from workflow_artifacts.lifecycle.screen_publication import ScreenArtifactPublicationCapability
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.common.sampling_plan import SeedRole, SelectionPlan
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest, ScreenRoundInputs, ScreenServices, execute_screen)
    from workflow_core.structures.formats import read_extxyz_structures
    from test_screen_executor import _Reporter
    roots = tuple(sorted((Path(root)/'tasks').glob('task_*')))
    assert roots
    source = GcmcObservationSource(tuple((r, validate_task(r)) for r in roots))
    records = source.catalog()
    assert records and any(r.phase == 'md' for r in records)
    plan = SelectionPlan.from_seeds(tuple(SeedRole(role, group) for role, group in
        dict.fromkeys((r.sampling_role_id, r.selection_group_id) for r in records)))
    schema = {'schema': 'readout-input@1', 'width': 2}
    manifest = dict(model_digest='explicit-interface-fixture',
        reference_digest='explicit-interface-fixture', descriptor_schema=schema)
    teacher_router = tmp_path/'teacher-router'
    teacher_router.mkdir()
    (teacher_router/'manifest.json').write_text(json.dumps(manifest))
    def score_teacher(*, frames, **kwargs):
        result = []
        for frame in frames:
            frame = frame.copy()
            frame.info['teacher_p'] = .2
            result.append(frame)
        return SimpleNamespace(frames=tuple(result), frame_means=None, descriptor_schema=schema)
    def score_student(*, frames, router_dir, **kwargs):
        router_dir.mkdir(parents=True)
        (router_dir/'manifest.json').write_text(json.dumps(manifest))
        return SimpleNamespace(values={f.info['structure_id']: .2 for f in frames},
            frame_means=None, descriptor_schema=schema)
    stage = tmp_path/'screen'
    stage.mkdir()
    result = execute_screen(ScreenExecutionRequest(
        route=resolve_route(teacher={'backend': 'MACE', 'model': 'single'},
            student={'backend': 'DeepMD', 'cli': 'pt', 'model': 'single'},
            sampler={'backend': 'LAMMPS'}),
        round_index=0, stage=stage, source=source,
        inputs=ScreenRoundInputs(system_ids=plan.group_ids, selection_plan=plan,
            require_explicit_provenance=True, pair_guard=None, screen_policy='random',
            fp_policy='random', pool=-1, keep=keep, fp_pool=-1,
            teacher_p={'floor': 0., 'trust': .1, 'ceiling': .3},
            student_p={'floor': 0., 'ceiling': .3}, entropy='saved-gpu-screen-interface'),
        services=ScreenServices(score_teacher=score_teacher, score_student=score_student,
            label_teacher=lambda *, frames, **kwargs: [f.copy() for f in frames]),
        report=_Reporter(), artifacts=ScreenArtifactPublicationCapability(),
        teacher_router=teacher_router))
    known = {r.structure_id for r in records}
    for path in (result.train_path, result.fp_candidate_path):
        frames = read_extxyz_structures(path, allow_empty=False)
        assert all(f.info['structure_id'] in known for f in frames)
        assert all('dice_atom_id' in f.arrays and 'dice_molecule_id' in f.arrays for f in frames)
    assert result.train_count > 0 and result.fp_candidate_count > 0
    assert (stage/'selection_group_outcomes.json').is_file()
    return {f.info['structure_id'] for f in read_extxyz_structures(result.train_path)}
