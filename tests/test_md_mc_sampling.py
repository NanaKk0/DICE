import json
from pathlib import Path

import pytest

from workflow_core.lifecycle.config import load_run_config
from test_sampling_conditions import _write_lammps_run


def write_run(tmp_path, *, steps=100000, interval=10, attempts=1, ensemble='nvt', pressures=(1.,)):
    path = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'round_count': 1, 'seeds': {'ysz': {'source': 'data/A.xyz'}},
        'condition_groups': {'swap': {
            'kind': 'md_mc', 'ensemble': ensemble, 'temperatures_k': [1773],
            **({'pressures_bar': list(pressures)} if ensemble == 'npt' else {}),
            'steps': steps, 'recipe': 'recipes/nvt.lammps',
            'mc': {'move': 'identity_swap', 'elements': ['Y', 'Zr'],
                   'every_md_steps': interval, 'attempts_per_event': attempts},
        }},
    })
    payload = json.loads(path.read_text())
    payload['teacher']['type_map'] = payload['student']['type_map'] = ['O', 'Y', 'Zr']
    path.write_text(json.dumps(payload))
    return path


def test_user_can_declare_nvt_identity_swap_budget(tmp_path):
    plan = load_run_config(write_run(tmp_path)).sampling_plan
    task, = plan.tasks_for_round(0)
    assert (task.kind, task.ensemble, task.temperature_k, task.steps) == ('md_mc', 'nvt', 1773, 100000)
    assert task.md_mc.planned_attempts(task.steps) == 10000
    assert task.md_mc.elements == ('Y', 'Zr')


@pytest.mark.parametrize('change', [
    {'ensemble': 'npt'}, {'mc': {'move': 'transmute'}},
    {'mc': {'elements': ['Y', 'Y']}}, {'mc': {'elements': ['Y', 'H']}},
    {'mc': {'every_md_steps': True}}, {'mc': {'attempts_per_event': 0}},
    {'mc': {'extra': 1}}, {'steps': 9}, {'temperatures_k': [float('nan')]},
])
def test_invalid_swap_declaration_is_rejected_without_native_work(tmp_path, change):
    path = write_run(tmp_path)
    payload = json.loads(path.read_text())
    group = payload['sample']['condition_groups']['swap']
    for key, value in change.items():
        if key == 'mc':
            group['mc'].update(value)
        else:
            group[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_run_config(path)


def _run_swap_fixture(tmp_path, ensemble='nvt', pressure_mode='iso', native_exception=None, inline_comments=False, automatic=False, clean=False):
    from ase import Atoms
    from workflow_backends.lammps.sampler import LammpsSampler
    from workflow_backends.lammps.observation_source import LammpsObservationSource

    run = write_run(tmp_path, steps=20, ensemble=ensemble)
    recipe = tmp_path / 'recipes/nvt.lammps'
    recipe.write_text('''# DICE_CAPABILITY: md_mc
units metal
boundary p p p
atom_style atomic
variable dice_trajectory_every_steps equal 10
include dice_task_setup.lammps
timestep 0.001
velocity all create ${dice_temperature} ${dice_seed}
fix thermostat all nvt temp ${dice_temperature} ${dice_temperature} 0.1
include dice_md_mc.lammps
''')
    if ensemble == 'npt':
        recipe.write_text(recipe.read_text().replace('fix thermostat all nvt temp ${dice_temperature} ${dice_temperature} 0.1',
            f'fix thermostat all npt temp ${{dice_temperature}} ${{dice_temperature}} 0.1 {pressure_mode} ${{dice_pressure}} ${{dice_pressure}} 1.0 flip no'))
    if inline_comments:
        recipe.write_text(recipe.read_text().replace('equal 10\n', 'equal 10 # cadence\n').replace(
            'timestep 0.001\n', 'timestep 0.001 # 1 fs\n'))
    if automatic:
        recipe.write_text(recipe.read_text().replace('include dice_task_setup.lammps\n', ''))
    if clean:
        recipe.write_text('units metal\nboundary p p p\natom_style atomic\ntimestep 0.001\n'
            'velocity all create 300 917\n'
            'dump extra all custom 10 extra.dump id type x y z\n'
            f'fix thermostat all {ensemble} temp 300 300 0.1'+
            (f' {pressure_mode} 1 1 1.0 flip no' if ensemble=='npt' else '')+'\nrun 20\n')
    plan = load_run_config(run).sampling_plan
    model = tmp_path / 'model.pb'
    model.write_text('model')
    calls = []

    class Executor:
        def run(self, task_dir, task):
            calls.append(task['kind'])
            if native_exception is not None:
                raise native_exception
            (task_dir / 'traj.lammpstrj').write_text(''.join(
                f'ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n3\n'
                'ITEM: BOX BOUNDS pp pp pp\n'+(f'0 {5+s*.001}\n'*3 if ensemble == 'npt' else '0 5\n'*3)+
                'ITEM: ATOMS id type x y z\n1 1 0 0 0\n'
                f'2 {a} 1 1 1\n3 {b} 2 2 2\n'
                for s, a, b in [(10, 3, 2), (20, 2, 3)]))
            (task_dir / 'mc_stats.out').write_text('10 1 1\n20 2 2\n')
            (task_dir / 'mc_terminal.out').write_text('DICE_MD_MC_COMPLETE 20 2 2\n')

    seed = Atoms('OYZr', positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[5]*3, pbc=True)
    LammpsSampler(run, 0, 0, Executor()).sample(
        group_recipes={'swap': recipe}, sample_dir=tmp_path / 'sample',
        models=(model,), type_map=('O', 'Y', 'Zr'), systems={'ysz': [seed]},
        steps=None, plan=plan, plumed_sources={},
    )
    assert calls == ['md_mc']
    source = LammpsObservationSource(sample_dir=tmp_path/'sample', run_json_path=run,
        iter_index=0, round_index=0, type_map=('O', 'Y', 'Zr'), model_count=1, plan=plan,
        accepted_kinds=('md_mc',))
    frames = [f for chunk in source.replay([r.structure_id for r in source.catalog()], chunk_size=1) for f in chunk]
    assert [f.get_chemical_symbols() for f in frames] == [['O', 'Zr', 'Y'], ['O', 'Y', 'Zr']]
    assert [f.info['mc_attempts'] for f in frames] == [1, 2]
    assert all('force_uncertainty' not in frame.info and 'force_std_max' not in frame.info for frame in frames)
    return frames


def test_swap_task_runs_once_and_replays_changed_species(tmp_path):
    _run_swap_fixture(tmp_path)


def test_md_mc_sparse_replay_does_not_rescan_committed_history(tmp_path, monkeypatch):
    from workflow_backends.lammps.observation_source import LammpsObservationSource
    _run_swap_fixture(tmp_path)
    source = LammpsObservationSource(sample_dir=tmp_path/'sample', run_json_path=tmp_path/'run.json',
        iter_index=0, round_index=0, type_map=('O','Y','Zr'), model_count=1,
        plan=load_run_config(tmp_path/'run.json').sampling_plan, accepted_kinds=('md_mc',))
    rows = source.catalog()
    trajectory = tmp_path/'sample/tasks/task_000000/md_mc_frames.lammpstrj'
    original_open, read_bytes = Path.open, []
    class Counted:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def __getattr__(self, name): return getattr(self.stream, name)
        def read(self, *args):
            value=self.stream.read(*args);read_bytes.append(len(value));return value
        def readline(self, *args):
            value=self.stream.readline(*args);read_bytes.append(len(value));return value
    def opened(path,*args,**kwargs):
        stream=original_open(path,*args,**kwargs)
        return Counted(stream) if path==trajectory else stream
    monkeypatch.setattr(Path,'open',opened)
    frame, = next(source.replay((rows[-1].structure_id,),chunk_size=1))
    assert frame.info['mc_attempts'] == 2
    assert frame.info['phase'] == 'md_mc'
    assert sum(read_bytes) < trajectory.stat().st_size * .75
    trajectory.write_text(trajectory.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        tuple(source.replay((rows[-1].structure_id,),chunk_size=1))


@pytest.mark.parametrize('ensemble',['nvt','npt'])
def test_swap_accepts_normal_native_protocol_without_markers(tmp_path,ensemble):
    _run_swap_fixture(tmp_path,ensemble=ensemble,clean=True)
    text=(tmp_path/'sample/tasks/task_000000/input.lammps').read_text()
    assert 'velocity all create 300 917' in text
    assert f'fix thermostat all {ensemble} temp ${{dice_temperature}}' in text


def test_committee_worker_evaluates_only_formal_frames(tmp_path):
    import numpy as np
    from workflow_backends.lammps.md_mc_evaluation import evaluate_task
    from workflow_backends.lammps.identity_swap import validate_result
    from workflow_backends.sample_artifacts import load_task_context, write_task_context
    from dataclasses import replace

    _run_swap_fixture(tmp_path)
    root = tmp_path/'sample/tasks/task_000000'
    context = load_task_context(root/'task_context.json')
    model = root/'graph_001.pb'
    model.write_bytes((root/'graph_000.pb').read_bytes())
    context = replace(context, ordered_models=('graph_000.pb', 'graph_001.pb'),
        model_digests=context.model_digests*2,
        md_mc={**context.md_mc, 'member_ids': ['student_01', 'student_02']})
    write_task_context(root/'task_context.json', context)
    # Native trial output is irrelevant to the retained-frame evaluator.
    (root/'model_devi.out').write_text('10 0 0 0 100 0 0\n10 0 0 0 999 0 0\n')
    seen = []

    class Evaluator:
        def evaluate(self, frames):
            for frame in frames:
                seen.append(frame.get_chemical_symbols())
                yield np.array([1., 2.]), np.array([np.zeros((3, 3)), np.ones((3, 3))*2])
        def close(self):
            pass

    class Provider:
        def open(self, paths, symbols):
            assert [p.name for p in paths] == ['graph_000.pb', 'graph_001.pb']
            return Evaluator()

    evaluate_task(root, provider=Provider())
    _, result = validate_result(root)
    assert result['status'] == 'completed'
    assert seen == [['O', 'Zr', 'Y'], ['O', 'Y', 'Zr']]
    assert [r['uncertainty'] for r in result['observations']] == pytest.approx([np.sqrt(3)]*2)


def test_failed_prefix_keeps_failed_status_and_cannot_skip_bad_audit(tmp_path):
    from workflow_backends.lammps.identity_swap import finalize, validate_result
    from workflow_backends.lammps.observation_source import LammpsObservationSource
    _run_swap_fixture(tmp_path)
    root = tmp_path/'sample/tasks/task_000000'
    (root/'mc_stats.out').write_text('10 1 1\n20 999 2\n')
    finalize(root, failure='native failed')
    _, result = validate_result(root)
    assert result['status'] == 'failed'
    assert result['completed_steps'] is None
    assert result['actual_attempts'] is None
    assert [r['step'] for r in result['observations']] == [10]
    source = LammpsObservationSource(sample_dir=tmp_path/'sample', run_json_path=tmp_path/'run.json',
        iter_index=0, round_index=0, type_map=('O', 'Y', 'Zr'), model_count=1)
    assert source.failed_task_count == 1
    assert [r.md_step for r in source.catalog()] == [10]


@pytest.mark.parametrize('ensemble', ['nvt', 'npt'])
def test_mixed_md_mc_multigpu_pool_preserves_slots_and_drains_after_failure(tmp_path, monkeypatch, ensemble):
    import threading
    from contextlib import contextmanager
    import numpy as np
    from ase import Atoms
    from workflow_backends.execution import current_operation_environment_overrides
    from workflow_backends.lammps.sampler import LammpsSampler
    from workflow_backends.lammps.observation_source import LammpsObservationSource
    from workflow_backends.lammps.identity_swap import finalize
    from workflow_runtime.execution.allocation import Allocation
    from workflow_runtime.lifecycle.environment import OperationContext, LammpsExecutionPolicy

    _run_swap_fixture(tmp_path)
    run = tmp_path/'run.json'
    payload = json.loads(run.read_text())
    payload['sample']['condition_groups'] = {
        'md': {'kind': 'nvt', 'temperatures_k': [1700, 1773], 'steps': 20, 'recipe': 'recipes/md.in'},
        'swap': {**payload['sample']['condition_groups']['swap'], 'temperatures_k': [1700, 1773, 1800], 'steps': 20},
    }
    if ensemble == 'npt':
        payload['sample']['condition_groups']['swap'].update(ensemble='npt', pressures_bar=[1.])
        recipe = tmp_path/'recipes/nvt.lammps'
        recipe.write_text(recipe.read_text().replace('fix thermostat all nvt temp ${dice_temperature} ${dice_temperature} 0.1',
            'fix thermostat all npt temp ${dice_temperature} ${dice_temperature} 0.1 iso ${dice_pressure} ${dice_pressure} 1.0 flip no'))
    run.write_text(json.dumps(payload))
    md = tmp_path/'recipes/md.in'
    md.write_text('variable dice_trajectory_every_steps equal 10\ninclude dice_task_setup.lammps\nrun ${dice_run_steps}\n')
    # This test exercises the historical internal sampler queue; clean protocols
    # have their own public native adapter and pool qualification.
    from dataclasses import replace
    plan = load_run_config(run).sampling_plan
    plan = replace(plan, condition_groups=tuple(replace(group, protocol_request=None)
        for group in plan.condition_groups))
    model = tmp_path/'model.pb'
    model2 = tmp_path/'model2.pb'
    model2.write_text('second model')
    barrier = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    assignments, evaluated, sessions = {}, {}, []
    active = {'0': 0, '1': 0}
    maximum = dict(active)

    class Evaluator:
        def evaluate(self, frames):
            for frame in frames:
                yield np.array([0., 1.]), np.array([np.zeros((len(frame), 3)), np.ones((len(frame), 3))])

    class Executor:
        context = OperationContext(role='sampler.lammps.tf', profile_id='test', source_scripts=(),
            executables={}, runtime={}, allocation=Allocation(1, 2, 2, 2048, 'direct', 'pool', 2))
        execution_policy = LammpsExecutionPolicy(2, 50)

        @contextmanager
        def open_mps_device_sessions(self, root, devices):
            sessions.extend(('start', d) for d in devices)
            try:
                yield {d: {'CUDA_MPS_PIPE_DIRECTORY': '/mps/'+d} for d in devices}
            finally:
                sessions.extend(('stop', d) for d in devices)

        def run(self, root, task, **kwargs):
            env = current_operation_environment_overrides()
            device = env['CUDA_VISIBLE_DEVICES']
            index = int(root.name.rsplit('_', 1)[1])
            with lock:
                assignments[index] = dict(env)
                active[device] += 1
                maximum[device] = max(maximum[device], active[device])
            if index < 4:
                barrier.wait()
            steps = [10] if index == 2 else [10, 20]
            (root/'traj.lammpstrj').write_text(''.join(
                f'ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n3\n'
                'ITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n'
                'ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 2 1 1 1\n3 3 2 2 2\n'
                for s in steps))
            if task['kind'] == 'md_mc':
                (root/'mc_stats.out').write_text(''.join(f'{s} {s//10} 0\n' for s in steps))
                if index != 2:
                    (root/'mc_terminal.out').write_text('DICE_MD_MC_COMPLETE 20 2 0\n')
            else:
                (root/'model_devi.out').write_text('10 0 0 0 0.2 0 0\n20 0 0 0 0.2 0 0\n')
            with lock:
                active[device] -= 1
            if index == 2:
                raise RuntimeError('controlled native failure')

        def finalize_md_mc(self, root, *, failure=None):
            evaluated[int(root.name.rsplit('_', 1)[1])] = dict(current_operation_environment_overrides())
            finalize(root, failure=failure, evaluator=Evaluator())

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    seed = Atoms('OYZr', positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[5]*3, pbc=True)
    LammpsSampler(run, 0, 0, Executor()).sample(
        group_recipes={'md': md, 'swap': tmp_path/'recipes/nvt.lammps'}, sample_dir=tmp_path/'mixed',
        models=(model, model2), type_map=('O', 'Y', 'Zr'), systems={'ysz': [seed]}, steps=None, plan=plan, plumed_sources={})
    assert maximum == {'0': 2, '1': 2}
    assert sorted(assignments) == list(range(5))
    assert evaluated == {i: assignments[i] for i in (2, 3, 4)}
    assert sessions == [('start', '0'), ('start', '1'), ('stop', '0'), ('stop', '1')]
    source = LammpsObservationSource(sample_dir=tmp_path/'mixed', run_json_path=run,
        iter_index=0, round_index=0, type_map=('O', 'Y', 'Zr'), model_count=2, plan=plan)
    assert source.failed_task_count == 1
    assert [(r.task_index, r.md_step) for r in source.catalog()] == [
        (0, 10), (0, 20), (1, 10), (1, 20), (2, 10), (3, 10), (3, 20), (4, 10), (4, 20)]


def test_committed_md_mc_is_consumer_runnable_without_native_inference(tmp_path, monkeypatch):
    from workflow_backends.sample_recovery import validate_sample_artifacts
    from workflow_core.lifecycle.sampling import SampleRecoveryFacts, SamplingTaskIntent
    from workflow_backends.widom.evaluation import DeepMDEvaluationProvider
    test_committee_worker_evaluates_only_formal_frames(tmp_path)
    condition, = load_run_config(tmp_path/'run.json').sampling_plan.tasks
    intent = SamplingTaskIntent(condition.task_id, condition.system_id, condition.ensemble,
        condition.temperature_k, condition_group_id=condition.condition_group_id,
        sampling_role_id=condition.sampling_role_id, selection_group_id=condition.selection_group_id,
        kind=condition.kind, steps=condition.steps, md_mc=condition.md_mc)
    facts = SampleRecoveryFacts('LAMMPS', 'tf', ('student_01', 'student_02'), ('O', 'Y', 'Zr'),
        ('ysz',), (intent,), None, False, False)
    monkeypatch.setattr(DeepMDEvaluationProvider, 'open', lambda *args: pytest.fail('read-only validation loaded a model'))
    assert validate_sample_artifacts(tmp_path/'sample', facts)
    (tmp_path/'sample/tasks/task_000000/md_mc_members.jsonl').write_text('corrupt\n')
    assert not validate_sample_artifacts(tmp_path/'sample', facts)


@pytest.mark.parametrize('damage', ['id', 'composition', 'oxygen', 'cell', 'cadence', 'coordinate'])
def test_malformed_geometry_stops_prefix_at_first_invalid_frame(tmp_path, damage):
    from workflow_backends.lammps.identity_swap import finalize, native_blocks, validate_result
    _run_swap_fixture(tmp_path)
    root = tmp_path/'sample/tasks/task_000000'
    first, second = list(native_blocks(root/'traj.lammpstrj'))
    replacements = {
        'id': ('3 3 2 2 2', '2 3 2 2 2'),
        'composition': ('2 2 1 1 1', '2 3 1 1 1'),
        'oxygen': ('1 1 0 0 0\n2 2 1 1 1', '1 2 0 0 0\n2 1 1 1 1'),
        'cell': ('0 5\n', '0 6\n'),
        'cadence': ('TIMESTEP\n20\n', 'TIMESTEP\n30\n'),
        'coordinate': ('2 2 1 1 1', '2 2 nan 1 1'),
    }
    old, new = replacements[damage]
    assert old in second
    (root/'traj.lammpstrj').write_text(first+second.replace(old, new, 1))
    finalize(root)
    _, result = validate_result(root)
    assert result['status'] == 'failed'
    assert [row['step'] for row in result['observations']] == [10]
    assert result['acceptance_ratio'] is None
    assert result['planned_attempts'] == 2


def test_canonical_audit_rejects_boolean_counter(tmp_path):
    from workflow_backends.lammps.identity_swap import validate_result
    _run_swap_fixture(tmp_path)
    root = tmp_path/'sample/tasks/task_000000'
    path = root/'md_mc_result.json'
    result = json.loads(path.read_text())
    result['observations'][0]['accepts'] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='integers'):
        validate_result(root)


def test_selected_suffix_preflight_does_not_open_inactive_recipe(tmp_path):
    from types import SimpleNamespace
    from workflow_app.composition import preflight_sampling_protocols
    _run_swap_fixture(tmp_path)
    path = tmp_path/'run.json'
    payload = json.loads(path.read_text())
    sample = payload['sample']
    sample.pop('round_count')
    sample['condition_groups']['md'] = {'kind': 'nvt', 'temperatures_k': [1773], 'steps': 20, 'recipe': 'recipes/md.in'}
    sample['rounds'] = [{'condition_groups': ['swap']}, {'condition_groups': ['md']}]
    (tmp_path/'recipes/md.in').write_text(
        'units metal\natom_style atomic\ntimestep 0.001\n'
        'fix bath all nvt temp ${dice_temperature} ${dice_temperature} 0.1\n'
        'run ${dice_run_steps}\n')
    path.write_text(json.dumps(payload))
    launch = SimpleNamespace(config=load_run_config(path), preflight=None,
        plan=SimpleNamespace(blocks=tuple(SimpleNamespace(block_id=block) for block in ('r0/sample', 'r0/screen', 'r1/sample', 'fp'))))
    (tmp_path/'recipes/nvt.lammps').write_text('run 999\n')
    preflight_sampling_protocols(launch, first_block='r1/sample')
    preflight_sampling_protocols(launch, first_block='fp')
    with pytest.raises(ValueError, match='thermostat'):
        preflight_sampling_protocols(launch, first_block='r0/sample')


@pytest.mark.parametrize('extra', ['units real', 'boundary f f f', 'atom_style full', 'run 20', 'reset_timestep 0'])
def test_managed_recipe_rejects_native_escape_commands(tmp_path, extra):
    from workflow_backends.lammps.identity_swap import validate_recipe
    _run_swap_fixture(tmp_path)
    text = (tmp_path/'recipes/nvt.lammps').read_text()
    with pytest.raises(ValueError):
        validate_recipe(text.replace('include dice_md_mc.lammps', extra+'\ninclude dice_md_mc.lammps'))


@pytest.mark.parametrize('mutation', ['left_handed', 'isotope'])
def test_seed_rejects_unrepresented_cell_or_masses(mutation):
    from ase import Atoms
    from workflow_core.common.md_mc import IdentitySwapParameters
    from workflow_core.structures.md_mc import validate_seed
    seed = Atoms('YZrO', positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[5]*3, pbc=True)
    if mutation == 'left_handed':
        seed.cell[0] *= -1
    else:
        seed.set_masses(seed.get_masses()*2)
    with pytest.raises(ValueError):
        validate_seed(seed, IdentitySwapParameters(('Y', 'Zr'), 10, 1), ('O', 'Y', 'Zr'))


def test_npt_identity_swap_expands_temperature_pressure_product(tmp_path):
    path = write_run(tmp_path, steps=103, interval=10, attempts=2, ensemble='npt', pressures=(1., 100.))
    payload = json.loads(path.read_text())
    payload['sample']['condition_groups']['swap']['temperatures_k'] = [1500, 1773]
    path.write_text(json.dumps(payload))
    plan = load_run_config(path).sampling_plan
    assert plan.ensembles == ('npt',)
    assert [(task.ensemble, task.temperature_k, task.pressure_bar) for task in plan.tasks] == [
        ('npt', 1500., 1.), ('npt', 1500., 100.), ('npt', 1773., 1.), ('npt', 1773., 100.)]
    assert all(task.md_mc.planned_attempts(task.steps) == 20 for task in plan.tasks)


@pytest.mark.parametrize('pressure_mode', ['iso', 'aniso', 'tri'])
def test_npt_swap_replays_instantaneous_cell_and_pressure(tmp_path, pressure_mode):
    from workflow_backends.lammps.identity_swap import validate_result
    frames = _run_swap_fixture(tmp_path, ensemble='npt', pressure_mode=pressure_mode)
    assert [frame.cell[0, 0] for frame in frames] == pytest.approx([5.01, 5.02])
    assert [frame.info['ensemble'] for frame in frames] == ['npt', 'npt']
    assert [frame.info['pressure_bar'] for frame in frames] == [1., 1.]
    context, result = validate_result(tmp_path/'sample/tasks/task_000000')
    assert context.ensemble == 'npt' and result['status'] == 'completed'
    if pressure_mode == 'tri':
        assert 'change_box all triclinic' in (tmp_path/'sample/tasks/task_000000/dice_task_setup.lammps').read_text()


@pytest.mark.parametrize('pressure', [None, [], [True], [float('inf')], [1, 1]])
def test_npt_swap_rejects_missing_or_invalid_mechanical_pressure(tmp_path, pressure):
    path = write_run(tmp_path, ensemble='npt')
    payload = json.loads(path.read_text())
    group = payload['sample']['condition_groups']['swap']
    if pressure is None:
        group.pop('pressures_bar')
    else:
        group['pressures_bar'] = pressure
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_run_config(path)


def test_npt_recipe_cannot_be_used_for_nvt_or_escape_cell_policy(tmp_path):
    from workflow_backends.lammps.identity_swap import validate_recipe
    _run_swap_fixture(tmp_path, ensemble='npt')
    text = (tmp_path/'recipes/nvt.lammps').read_text()
    validate_recipe(text, ensemble='npt')
    for damaged in (text, text.replace('flip no', 'flip yes'), text.replace('1.0 flip no', '0 flip no'),
                    text.replace('iso ${dice_pressure} ${dice_pressure}', 'iso 1 1')):
        with pytest.raises(ValueError):
            validate_recipe(damaged, ensemble='nvt' if damaged == text else 'npt')


def test_npt_invalid_cell_stops_failed_prefix(tmp_path):
    from workflow_backends.lammps.identity_swap import native_blocks, finalize, validate_result
    _run_swap_fixture(tmp_path, ensemble='npt')
    root = tmp_path/'sample/tasks/task_000000'
    first, second = list(native_blocks(root/'traj.lammpstrj'))
    (root/'traj.lammpstrj').write_text(first+second.replace('0 5.02', '0 nan', 1))
    finalize(root, failure='controlled failure')
    _, result = validate_result(root)
    assert result['status'] == 'failed' and result['last_retained_step'] == 10


@pytest.mark.parametrize('ensemble', ['nvt', 'npt'])
def test_md_mc_user_cancellation_is_not_prefix_salvage(tmp_path, ensemble):
    with pytest.raises(KeyboardInterrupt, match='cancelled'):
        _run_swap_fixture(tmp_path, ensemble=ensemble, native_exception=KeyboardInterrupt('cancelled'))
    assert not (tmp_path/'sample/tasks/task_000000/md_mc_result.json').exists()


def test_npt_committee_inference_failure_keeps_only_matching_prefix(tmp_path):
    from dataclasses import replace
    import numpy as np
    from workflow_backends.sample_artifacts import load_task_context, write_task_context
    from workflow_backends.lammps.identity_swap import finalize, validate_result
    _run_swap_fixture(tmp_path, ensemble='npt')
    root = tmp_path/'sample/tasks/task_000000'
    context = load_task_context(root/'task_context.json')
    (root/'graph_001.pb').write_bytes((root/'graph_000.pb').read_bytes())
    context = replace(context, ordered_models=('graph_000.pb', 'graph_001.pb'), model_digests=context.model_digests*2,
        md_mc={**context.md_mc, 'member_ids': ['student_01', 'student_02']})
    write_task_context(root/'task_context.json', context)

    class Evaluator:
        def evaluate(self, frames):
            for frame in frames:
                if frame.info['md_step'] == 20:
                    raise RuntimeError('controlled inference failure')
                yield np.array([0., 1.]), np.array([np.zeros((3, 3)), np.ones((3, 3))])

    finalize(root, evaluator=Evaluator())
    _, result = validate_result(root)
    assert result['status'] == 'failed' and result['last_retained_step'] == 10
    assert result['actual_attempts'] is None


def test_both_complete_examples_and_managed_recipes_are_structurally_valid():
    from pathlib import Path
    from workflow_backends.lammps.identity_swap import validate_recipe
    root = Path(__file__).parent/'fixtures/md_mc_acquisition'
    for name, ensemble in (('run.json', 'nvt'), ('run_npt.json', 'npt')):
        config = load_run_config(root/name)
        task, = config.sampling_plan.tasks_for_round(0)
        assert task.ensemble == ensemble and task.md_mc.planned_attempts(task.steps) == 10000
        recipe = config.raw['sample']['condition_groups']['ysz_swap']['recipe']
        validate_recipe((root/recipe).read_text(), ensemble=ensemble)


@pytest.mark.parametrize('ensemble', ['nvt', 'npt'])
def test_managed_recipe_inline_comments_preserve_cadence_and_timestep(tmp_path, ensemble):
    frames = _run_swap_fixture(tmp_path, ensemble=ensemble, inline_comments=True)
    assert [frame.info['md_step'] for frame in frames] == [10, 20]
    assert [frame.info['md_time_fs'] for frame in frames] == [10., 20.]


@pytest.mark.parametrize('binding', ['temperature', 'pressure', 'driving_member'])
def test_consumer_rejects_context_native_binding_disagreement(tmp_path, binding):
    from dataclasses import replace
    from workflow_backends.sample_artifacts import load_task_context, write_task_context
    from workflow_backends.lammps.identity_swap import finalize, validate_result
    _run_swap_fixture(tmp_path, ensemble='npt')
    root = tmp_path/'sample/tasks/task_000000'
    context = load_task_context(root/'task_context.json')
    if binding == 'temperature':
        context = replace(context, temperature_k=1200.)
    elif binding == 'pressure':
        context = replace(context, pressure_bar=2.)
    else:
        (root/'graph_001.pb').write_bytes((root/'graph_000.pb').read_bytes())
        context = replace(context, ordered_models=('graph_001.pb',))
    write_task_context(root/'task_context.json', context)
    # Regenerate through the owning product writer, to test intrinsic semantic
    # disagreement rather than merely a stale context digest.
    finalize(root)
    with pytest.raises(ValueError, match='binding|conditions'):
        validate_result(root)


@pytest.mark.parametrize('ensemble', ['nvt', 'npt'])
def test_swap_protocol_automatically_assembles_task_setup(tmp_path, ensemble):
    _run_swap_fixture(tmp_path, ensemble=ensemble, automatic=True)


def test_swap_protocol_preflight_accepts_automatic_setup(tmp_path):
    from workflow_backends.lammps.identity_swap import validate_recipe
    recipe = (Path(__file__).parent / 'fixtures/md_mc_acquisition/recipes/ysz-nvt-swap.in').read_text()
    recipe = recipe.replace('include dice_task_setup.lammps\n', '')
    assert validate_recipe(recipe, ensemble='nvt')['ensemble'] == 'nvt'
