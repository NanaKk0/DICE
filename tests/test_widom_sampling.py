import pytest

from workflow_core.lifecycle.config import load_run_config
from test_sampling_conditions import _write_lammps_run


def test_public_widom_has_independent_host_budgets_without_md_controls(tmp_path):
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'round_count': 1,
        'seeds': {'A': {'source': 'data/A.xyz', 'selection_group': 'joint'},
                  'B': {'source': 'data/B.xyz', 'selection_group': 'joint'}},
        'condition_groups': {'insert': {'kind': 'widom', 'guest': 'data/guest.xyz',
                                        'frames': 3, 'max_attempts': 20}}})
    config = load_run_config(run)
    tasks = config.sampling_plan.tasks_for_round(0)
    assert len(tasks) == 2
    assert all(t.kind == 'widom' and t.steps is None and t.recipe_binding is None
               and t.temperature_k is None for t in tasks)
    assert all(t.widom.frames == 3 and t.widom.max_attempts == 20 for t in tasks)
    from workflow_app.composition import sampling_backend_binding
    binding = sampling_backend_binding(config)
    assert binding.guest_inputs == {'insert': tmp_path/'data/guest.xyz'}
    assert binding.recipes == ()


def test_independent_insertions_fill_target_or_exhaust_collision_budget():
    import numpy as np
    from ase import Atoms
    from workflow_core.common.widom import WidomSamplingParameters
    from workflow_core.structures.widom import insertion_candidates
    host = Atoms('He', positions=[[1, 1, 1]], cell=[[8, 0, 0], [2, 8, 0], [1, 2, 8]], pbc=True)
    guests = [Atoms('HH', positions=[[0, 0, 0], [0, 0, .7]]),
              Atoms('HH', positions=[[0, 0, 0], [0, 0, .8]])]
    events = list(insertion_candidates(host, guests, WidomSamplingParameters(3, 20, overlap_scale=None), entropy='test'))
    assert len(events) == 3
    for attempt, index, frame in events:
        assert len(frame) == 3
        np.testing.assert_array_equal(frame.positions[:1], host.positions)
        assert np.linalg.norm(frame.positions[2]-frame.positions[1]) == pytest.approx(.7 if index == 0 else .8)
    rejected = list(insertion_candidates(host, guests, WidomSamplingParameters(3, 7, overlap_scale=100), entropy='test'))
    assert len(rejected) == 7 and all(frame is None for _, _, frame in rejected)
    repeated = list(insertion_candidates(host, guests, WidomSamplingParameters(3, 20, overlap_scale=None), entropy='test'))
    for left, right in zip(events, repeated):
        np.testing.assert_array_equal(left[2].positions, right[2].positions)


@pytest.mark.parametrize('members', [1, 2])
def test_candidates_publish_and_replay_ordered_member_evidence(tmp_path, members):
    from ase import Atoms
    import numpy as np
    from workflow_backends.widom.adapter import WidomSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.widom import WidomSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    guest = tmp_path/'guest.xyz'
    write_extxyz_structures(guest, [Atoms('Ne')])
    model = tmp_path/'model.fixture'; model.write_text('analytic')
    class Evaluator:
        def evaluate(self, frames):
            return [(np.arange(members, dtype=float), np.arange(members*len(f)*3).reshape(members,len(f),3)/7) for f in frames]
        def close(self): pass
    class Provider:
        def preflight(self, paths, symbols): pass
        def open(self, paths, symbols): return Evaluator()
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (SamplingConditionGroup(
        'insert', 'widom', ('A',), widom=WidomSamplingParameters(3, 10, overlap_scale=None)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
        (SamplingSeedSource('A', (Atoms('He', cell=[8]*3, pbc=True),)),),
        OrderedSamplingModels((model,)*members, tuple(f'member-{i}' for i in range(members))), None)
    backend = WidomSamplingNativeOps(provider=Provider(), guest_sources={'insert': guest}, type_map=('He','Ne'))
    assert backend.execute(SamplingExecutionRequest(evidence)).task_count == 1
    assert backend.validate_current(evidence).complete
    source = backend.open_current(evidence).source
    records = source.catalog()
    assert len(records) == 3
    assert all((r.student_uncertainty is None) == (members == 1) and r.md_step is None for r in records)
    ids = tuple(r.structure_id for r in records)[::-1]
    frames = [f for chunk in source.replay(ids, chunk_size=2) for f in chunk]
    assert [f.info['structure_id'] for f in frames] == list(ids)
    assert [f.info['proposal_index'] for f in frames] == [3, 2, 1]
    assert all(len(f) == 2 and 'dice_force_member_000' in f.arrays for f in frames)


@pytest.mark.parametrize('teacher_model,student_model', [('single','single'), ('single','committee'), ('committee','single'), ('committee','committee')])
@pytest.mark.parametrize('mode,policy', [('normal','random'), ('zero','random'), ('zero','ELEMENT_DIRECT'), ('failure','random'), ('interrupt','random')])
def test_public_widom_reaches_screen_and_fp(tmp_path, mode, teacher_model, student_model, policy, complex_plan=False):
    import json
    import numpy as np
    from dataclasses import replace
    from ase import Atoms
    from test_lifecycle_executor import _write_public_run, _FakeTeacher, _FakeStudent
    from lifecycle_role_fakes import compose_test_backend_roles
    from workflow_core.lifecycle.entrypoint import main
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from workflow_core.structures.formats import write_extxyz_structures, read_extxyz_structures
    from workflow_backends.widom.adapter import WidomSamplingNativeOps
    run = _write_public_run(tmp_path, teacher_model=teacher_model, student_model=student_model)
    write_extxyz_structures(tmp_path/'data/seed.xyz', [Atoms('H', cell=[8]*3, pbc=True)])
    write_extxyz_structures(tmp_path/'data/guest.xyz', [Atoms('H')])
    payload = json.loads(run.read_text()); payload['visualize'] = False
    payload['sampler'] = {'backend': 'LAMMPS'}
    payload['sample'] = {'round_count': 1, 'seeds': {'host': {'source': 'data/seed.xyz'}},
        'condition_groups': {'insert': {'kind': 'widom', 'guest': 'data/guest.xyz',
            'frames': 3, 'max_attempts': 10, 'overlap': {'scale': 100} if mode == 'zero' else None}}}
    if complex_plan:
        write_extxyz_structures(tmp_path/'data/seed.xyz', [Atoms('H',positions=[[x,0,0]],cell=[8]*3,pbc=True) for x in (0.,.2,.4)])
        payload['sample']['seeds'] = {
            'first': {'source':'data/seed.xyz','max_frames':2,'selection_group':'joint'},
            'second': {'source':'data/seed.xyz','selection_group':'joint'}}
        payload['sample']['condition_groups']['other'] = dict(payload['sample']['condition_groups']['insert'],random_seed=7)
        del payload['sample']['round_count']
        payload['sample']['rounds'] = [{'condition_groups':['insert']},{'seeds':['first'],'condition_groups':['other']}]
    payload['screen'].update(pool=3, keep=1, policy=policy)
    if policy in {'JOINT_DIRECT','ELEMENT_JOINT_DIRECT'}:
        payload['screen'].update(policy='random',prescreen=policy)
    if student_model == 'committee': payload['screen']['lo'] = 0.01
    run.write_text(json.dumps(payload))
    class Evaluator:
        calls = 0
        def evaluate(self, frames):
            self.calls += 1
            if self.calls == 2 and provider.fail:
                if mode == 'interrupt': raise KeyboardInterrupt('test interruption')
                raise ValueError('test inference failure')
            return [(np.zeros(1 if student_model == 'single' else 2), np.zeros((1, len(f), 3)) if student_model == 'single' else np.stack([np.zeros((len(f),3)), np.full((len(f),3), .17)])) for f in frames]
        def close(self): pass
    class Provider:
        fail = mode in {'failure', 'interrupt'}
        def preflight(self, paths, symbols): pass
        def open(self, paths, symbols): return Evaluator()
    provider = Provider()
    base = compose_test_backend_roles(run, teacher=_FakeTeacher(), student=_FakeStudent())
    native = WidomSamplingNativeOps(provider=provider, guest_sources={k: tmp_path/'data/guest.xyz' for k in payload['sample']['condition_groups']}, type_map=('H',))
    composition = replace(base, sampling=LifecycleSamplingCoordinator(native, base.student.sampling_models))
    if mode == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            main([str(run)], backend_composition=composition)
        provider.fail = False
        main([str(run), '--resume', 'r0/sample'], backend_composition=composition)
    else:
        main([str(run)], backend_composition=composition)
    selected = read_extxyz_structures(tmp_path/'iter0/fp/selected.xyz', allow_empty=True)
    assert bool(selected) == (mode != "zero" and teacher_model == "single")
    assert all(len(f) == 2 for f in selected)
    assert all('proposal_index' in f.info for f in selected)
    main([str(run)], backend_composition=composition)


def test_selected_environment_worker_executes_widom_without_md(tmp_path, monkeypatch):
    import numpy as np
    from ase import Atoms
    from types import SimpleNamespace
    from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory
    from workflow_backends.sampling import SamplingBackendBinding, SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.widom import WidomSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    from workflow_runtime.lifecycle.environment import OperationContext
    from workflow_backends.widom.worker import execute_manifest
    import workflow_backends.widom.runtime as runtime
    guest = tmp_path/'guest.xyz'; write_extxyz_structures(guest, [Atoms('H')])
    model = tmp_path/'model.fixture'; model.write_text('analytic')
    class Evaluator:
        def evaluate(self, frames):
            return [(np.zeros(1), np.zeros((1,len(f),3))) for f in frames]
        def close(self): pass
    class Provider:
        def preflight(self, paths, symbols): pass
        def open(self, paths, symbols): return Evaluator()
    context = OperationContext('sampler.lammps.tf', 'test', (), {'python': '/test/python'}, {})
    calls = []
    def operation(selected, executable, args, **kwargs):
        assert selected == context and executable == 'python'
        assert args[:3] == ('-m', 'workflow_backends.widom.worker', '--manifest')
        calls.append(args)
        execute_manifest(args[-1], provider=Provider())
    monkeypatch.setattr(runtime, 'run_operation', operation)
    backend = LammpsSamplingBackendFactory(SimpleNamespace(context=context)).bind(
        SamplingBackendBinding((), ('H',), {}, guest_inputs={'insert':guest}, widom_group_ids=('insert',)))
    plan = ExplicitSamplingPlan((SeedRole('A','A'),), (SamplingConditionGroup('insert','widom',('A',),
        widom=WidomSamplingParameters(2,10,overlap_scale=None)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json',0,0,tmp_path/'sample',plan,
        (SamplingSeedSource('A',(Atoms('H',cell=[8]*3,pbc=True),)),),OrderedSamplingModels((model,),('m0',)),None)
    backend.execute(SamplingExecutionRequest(evidence))
    assert len(calls) == 1
    assert len(backend.open_current(evidence).source.catalog()) == 2


@pytest.mark.parametrize('change', [{'frames':True}, {'max_attempts':2}, {'random_seed':-1},
    {'temperatures_k':[300]}, {'every':1}, {'overlap':{'scale':0}}, {'overlap':{'scale':None}}, {'overlap':{'radii_a':{}}}, {'overlap':{'radii_a':{'H':float('nan')}}}])
def test_invalid_public_widom_controls_fail(tmp_path, change):
    declaration = dict(kind='widom', guest='data/guest.xyz', frames=3, max_attempts=10)
    declaration.update(change)
    run = _write_lammps_run(tmp_path,sampler={'backend':'LAMMPS'},sample={'round_count':1,
        'seeds':{'A':{'source':'data/A.xyz'}},'condition_groups':{'insert':declaration}})
    with pytest.raises(ValueError): load_run_config(run)


def _bounded_backend(tmp_path, *, roles=('A',), provider=None, parameters=None):
    import numpy as np
    from ase import Atoms
    from workflow_backends.widom.adapter import WidomSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.widom import WidomSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    guest = tmp_path/'guest.xyz'
    write_extxyz_structures(guest, [Atoms('HH', positions=[[0,0,0],[0,0,.7]]),
                                   Atoms('HH', positions=[[0,0,0],[0,0,.9]])])
    model = tmp_path/'model.pb'; model.write_text('analytic test model')
    class Evaluator:
        def evaluate(self, frames): return [(np.zeros(1), np.zeros((1,len(f),3))) for f in frames]
        def close(self): pass
    class Provider:
        def preflight(self, paths, symbols): pass
        def open(self, paths, symbols): return Evaluator()
    plan = ExplicitSamplingPlan(tuple(SeedRole(r,'joint') for r in roles), (
        SamplingConditionGroup('insert','widom',roles,widom=parameters or WidomSamplingParameters(3,8,overlap_scale=None)),),1)
    host = Atoms('He',positions=[[1,1,1]],cell=[[8,0,0],[2,8,0],[1,2,8]],pbc=True)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json',0,0,tmp_path/'sample',plan,
        tuple(SamplingSeedSource(r,(host.copy(),)) for r in roles),OrderedSamplingModels((model,),('student',)),None)
    return WidomSamplingNativeOps(provider=provider or Provider(),guest_sources={'insert':guest},type_map=('He','H')), evidence


@pytest.mark.parametrize('change', ['positions','member_force','task_role','guest_index','model','counts'])
def test_committed_candidate_damage_is_rejected(tmp_path, change):
    import json
    from ase.io import read, write
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.widom.adapter import digest
    backend, evidence = _bounded_backend(tmp_path)
    backend.execute(SamplingExecutionRequest(evidence))
    root = evidence.sample_dir/'tasks/task_000000'
    metadata = root/'widom_result.json'; result = json.loads(metadata.read_text())
    if change in {'positions','member_force'}:
        frames = read(root/'observations.extxyz', ':')
        if change == 'positions': frames[0].positions[0,0] += 1
        else: frames[0].arrays['dice_force_member_000'][0,0] = float('nan')
        write(root/'observations.extxyz', frames)
        result['files']['observations.extxyz'] = digest(root/'observations.extxyz')
    elif change == 'task_role': result['observations'][0]['sampling_role_id'] = 'wrong'
    elif change == 'guest_index': result['observations'][0]['guest_index'] = 100
    elif change == 'counts': result['counts']['attempted'] = True
    else: (root/'graph_000.pb').write_text('damaged')
    metadata.write_text(json.dumps(result))
    assert not backend.validate_current(evidence).complete


def test_inference_failure_preserves_prefix_and_sibling_and_ignores_diagnostics(tmp_path):
    import numpy as np
    import json
    from workflow_backends.sampling import SamplingExecutionRequest
    class Evaluator:
        calls = 0
        def evaluate(self, frames):
            self.calls += 1
            f = frames[0]
            return [(np.zeros(1), np.full((1,len(f),3), np.nan if self.calls == 2 else 0.))]
        def close(self): pass
    class Provider:
        def preflight(self, paths, symbols): pass
        def open(self, paths, symbols): return Evaluator()
    backend, evidence = _bounded_backend(tmp_path, roles=('A','B'), provider=Provider())
    outcome = backend.execute(SamplingExecutionRequest(evidence))
    assert outcome.task_count == outcome.failed_task_count == 2
    assert len(backend.open_current(evidence).source.catalog()) == 2
    for task in evidence.plan.tasks_for_round(0):
        root = evidence.sample_dir/'tasks'/task.task_id
        result = json.loads((root/'widom_result.json').read_text())
        assert result['counts'] == dict(attempted=2, geometry_rejected=0, geometry_accepted=2, emitted=1)
        assert result['termination'] == 'failure'
        (root/'attempt.jsonl').unlink()
    assert backend.validate_current(evidence).complete


def test_process_death_salvages_only_complete_prefix(tmp_path):
    import numpy as np
    from workflow_backends.widom.adapter import validate_task
    class Evaluator:
        calls = 0
        def evaluate(self, frames):
            self.calls += 1
            if self.calls == 2: raise SystemExit('simulated worker death')
            return [(np.zeros(1), np.zeros((1,len(frames[0]),3)))]
        def close(self): pass
    class Provider:
        def open(self, paths, symbols): return Evaluator()
    backend, evidence = _bounded_backend(tmp_path, provider=Provider())
    task, seed, guests = backend.prepare(evidence)[0]
    (evidence.sample_dir/'tasks').mkdir(parents=True)
    root = evidence.sample_dir/'worker-staging'
    with pytest.raises(SystemExit): backend.execute_task(evidence,task,seed,guests,staging_root=root)
    with (root/'observations.extxyz').open('a') as stream: stream.write('12\ntruncated tail')
    backend.publish_process_failure(evidence,task,seed,guests,root,RuntimeError('worker exited'))
    result = validate_task(evidence.sample_dir/'tasks'/task.task_id)
    assert result['task_status'] == 'FAILED'
    assert result['counts'] == dict(attempted=2,geometry_rejected=0,geometry_accepted=2,emitted=1)


def test_collision_refill_matches_independent_mic_and_inclusive_boundary():
    import numpy as np
    from ase import Atoms
    from ase.geometry import find_mic
    from workflow_core.common.widom import WidomSamplingParameters
    from workflow_core.structures.widom import insertion_candidates
    host = Atoms('He',positions=[[1,1,1]],cell=[[3,0,0],[1,3,0],[.5,1,3]],pbc=True)
    guests = [Atoms('H')]
    proposals = list(insertion_candidates(host,guests,WidomSamplingParameters(30,30,overlap_scale=None),entropy='mic'))
    distances = [find_mic(f.positions[1]-host.positions[0],host.cell,True)[1] for _,_,f in proposals]
    scale = float(distances[0]/2)
    params = WidomSamplingParameters(5,30,overlap_scale=scale,radii_a=(('He',1.),('H',1.)))
    events = list(insertion_candidates(host,guests,params,entropy='mic'))
    expected = [i+1 for i,d in enumerate(distances) if d >= 2*scale][:5]
    assert [i for i,_,f in events if f is not None] == expected
    assert events[0][2] is not None  # equality is accepted
    assert len(events) == expected[-1]
    from dataclasses import replace
    scale = float(np.median(distances)/2)
    events = list(insertion_candidates(host,guests,replace(params,overlap_scale=scale),entropy='mic'))
    expected = [i+1 for i,d in enumerate(distances) if d >= 2*scale][:5]
    assert [i for i,_,f in events if f is not None] == expected
    assert len(events) == expected[-1] and len(events) > 5


def test_invalid_guest_does_not_destroy_previous_result(tmp_path):
    from workflow_backends.sampling import SamplingExecutionRequest
    backend,evidence = _bounded_backend(tmp_path)
    backend.execute(SamplingExecutionRequest(evidence))
    path = evidence.sample_dir/'tasks/task_000000/observations.extxyz'
    before = path.read_bytes()
    (tmp_path/'guest.xyz').write_text('invalid extxyz')
    with pytest.raises(Exception): backend.execute(SamplingExecutionRequest(evidence))
    assert path.read_bytes() == before


def test_mixed_widom_and_native_md_keep_one_catalog(tmp_path):
    from dataclasses import replace
    from workflow_backends.hybrid_sampling import WidomMixedNativeOps
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_core.common.sampling_plan import SamplingConditionGroup
    from test_sampling_backend_contract import _FakeLammpsExecutor
    widom,evidence = _bounded_backend(tmp_path)
    plan = replace(evidence.plan, rounds=None, condition_groups=(*evidence.plan.condition_groups,
        SamplingConditionGroup('md','nvt',('A',),steps_by_round=(20,),temperatures_k=(303.,))))
    evidence = replace(evidence,plan=plan)
    recipe = tmp_path/'md.in'; recipe.write_text('include dice_task_setup.lammps\nrun ${dice_run_steps}\n')
    backend = WidomMixedNativeOps(widom=widom,md=LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(),
        type_map=('He','H'),group_recipes={'md':recipe}))
    assert backend.execute(SamplingExecutionRequest(evidence)).task_count == 2
    assert backend.validate_current(evidence).complete
    records = backend.open_current(evidence).source.catalog()
    assert len(records) == 4 and [r.task_index for r in records] == [0,0,0,1]


@pytest.mark.parametrize('prescreen', ['JOINT_DIRECT','ELEMENT_JOINT_DIRECT'])
def test_zero_widom_population_supports_joint_prescreens(tmp_path, prescreen):
    # Reuse the actual public lifecycle with a single-model route.
    test_public_widom_reaches_screen_and_fp(tmp_path,'zero','single','single',prescreen)


def test_widom_rejects_nep_configuration(tmp_path):
    import json
    from pathlib import Path
    payload = json.loads(Path('tests/skills/dice_operator/fixtures/widom.json').read_text())
    payload['student'].update(backend='NEP',recipe='recipes/nep.in')
    del payload['student']['cli']
    payload['sampler']['backend'] = 'GPUMD'
    run = tmp_path/'run.json'; run.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='GPUMD'): load_run_config(run)


def test_bad_explicit_resume_input_keeps_published_suffix(tmp_path):
    import json
    from workflow_core.lifecycle.entrypoint import main
    test_public_widom_reaches_screen_and_fp(tmp_path,'normal','single','single','random')
    selected = tmp_path/'iter0/fp/selected.xyz'; original = selected.read_bytes()
    (tmp_path/'data/guest.xyz').write_text('broken input')
    # An ordinary completed invocation must not need the now-unused input.
    main([str(tmp_path/'run.json')])
    assert selected.read_bytes() == original
    with pytest.raises(ValueError): main([str(tmp_path/'run.json'),'--resume','r0/sample'])
    assert selected.read_bytes() == original



def test_source_frame_expansion_and_rounds_share_selection_quota(tmp_path):
    import json
    test_public_widom_reaches_screen_and_fp(tmp_path,'normal','single','single','random',complex_plan=True)
    for round_index, expected_tasks in [(0,3),(1,2)]:
        root = tmp_path/f'iter0/r{round_index}'
        results = [json.loads(p.read_text()) for p in sorted((root/'sample/tasks').glob('*/widom_result.json'))]
        assert len(results) == expected_tasks
        assert all(r['counts']['emitted'] == 3 and r['selection_group_id']=='joint' for r in results)
        outcomes = json.loads((root/'screen/selection_group_outcomes.json').read_text())
        assert len(outcomes['rows'])==1
        assert outcomes['rows'][0]['final_selected_count'] <= 1
        assert len({r['source_frame_index'] for r in results if 'first' in r['sampling_role_id']}) == 2


def test_atomic_publication_failure_requires_whole_task_rerun(tmp_path, monkeypatch):
    import workflow_backends.widom.adapter as adapter
    from workflow_backends.sampling import SamplingExecutionRequest
    backend,evidence = _bounded_backend(tmp_path)
    replace = adapter.os.replace
    captured = []
    def fail_publication(source, target):
        if target == evidence.sample_dir/'tasks/task_000000':
            captured.append((source/'observations.extxyz').read_bytes())
            raise OSError('injected publication failure')
        return replace(source,target)
    monkeypatch.setattr(adapter.os,'replace',fail_publication)
    with pytest.raises(OSError,match='publication'): backend.execute(SamplingExecutionRequest(evidence))
    assert not backend.validate_current(evidence).complete
    monkeypatch.setattr(adapter.os,'replace',replace)
    backend.execute(SamplingExecutionRequest(evidence))
    assert backend.validate_current(evidence).complete
    assert (evidence.sample_dir/'tasks/task_000000/observations.extxyz').read_bytes()==captured[0]
