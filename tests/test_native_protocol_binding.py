"""Clean native protocols inherit values and bind explicit scientific requests."""


def test_explicit_temperature_and_steps_preserve_initialization_and_native_method():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol

    text = ('units metal\natom_style atomic\ntimestep 0.002\n'
            'velocity mobile create 350 917 mom yes\n'
            'fix bath mobile nvt temp 1200 1200 0.15 drag 0.2\n'
            'compute disp mobile msd\nrun 999\n')
    protocol = compile_lammps_protocol(text, NativeProtocolRequest(temperature_k=600, steps=40))
    assert 'fix bath mobile nvt temp 600 600 0.15 drag 0.2' in protocol.text
    assert 'velocity mobile create 350 917 mom yes' in protocol.text
    assert 'compute disp mobile msd' in protocol.text
    assert 'run 40' in protocol.text
    assert protocol.ensemble == 'nvt'
    assert protocol.timestep_ps == .002
    assert protocol.settings['temperature_k'] == {'value': 600., 'source': 'DICE', 'target': 'bath'}
    assert protocol.settings['timestep_ps']['source'] == 'native'


def test_dice_completes_missing_controller_and_run_with_explicit_damping():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol
    request = NativeProtocolRequest(ensemble='npt', temperature_k=900, pressure_bar=2,
        steps=20, temperature_damping_ps=.15, pressure_damping_ps=1.5,
        pressure_coupling='aniso')
    result = compile_lammps_protocol('units metal\nvelocity all create 300 917\n', request)
    assert 'fix dice_integrator all npt temp 900 900 0.14999999999999999 aniso 2 2 1.5' in result.text
    assert 'run 20' in result.text
    assert 'velocity all create 300 917' in result.text
    assert result.total_steps == 20


def test_explicit_damping_and_group_bind_existing_native_controller_in_native_units():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol
    result = compile_lammps_protocol('units real\nfix bath all npt temp 300 300 100 iso 1 1 1000 drag .2\nrun 20\n',
        NativeProtocolRequest(group='mobile', temperature_damping_ps=.2,
            pressure_damping_ps=2, pressure_coupling='aniso'))
    assert 'fix bath mobile npt temp 300 300 200 aniso 1 1 2000 drag .2' in result.text
    assert result.settings['temperature_damping_ps']['source'] == 'DICE'


def test_recipe_only_inherits_native_variables_units_and_complete_run_budget():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol

    text = ('units real\nvariable target equal 300*2\nvariable count equal 25\n'
            'timestep 2\nfix bath all npt temp ${target} ${target} 100 x 2 2 1000 couple none\n'
            'run ${count}\nrun 15\n')
    protocol = compile_lammps_protocol(text, NativeProtocolRequest())
    assert protocol.text == text
    assert protocol.temperature_k == 600
    assert protocol.pressure_bar == 2 * 1.01325
    assert protocol.total_steps == 40
    assert protocol.timestep_ps == .002
    assert protocol.settings['temperature_k']['source'] == 'native'


def test_npt_overrides_preserve_pressure_axes_and_native_tail():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol

    text = ('units real\nfix bath mobile npt temp 300 300 100 '
            'x 1 1 1000 y 2 2 2000 couple none drag 0.3\nrun 50\n')
    result = compile_lammps_protocol(text, NativeProtocolRequest(pressure_bar=10.1325))
    assert 'x 10 10 1000 y 10 10 2000 couple none drag 0.3' in result.text
    assert 'temp 300 300 100' in result.text
    switched = compile_lammps_protocol(text, NativeProtocolRequest(ensemble='nvt', temperature_k=700))
    assert 'fix bath mobile nvt temp 700 700 100 drag 0.3' in switched.text
    assert switched.ensemble == 'nvt'
    assert switched.pressure_bar is None


def test_sampling_control_accepts_clean_options_without_a_mode():
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\ntimestep 0.001\n'
              'fix integrate all nve\nfix bath all langevin 300 300 0.2 12345 zero yes\nrun 80\n')
    result = render_lammps_control(source, {'trajectory_every_steps': 5},
        ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert 'fix bath all langevin 600 600 0.2 12345 zero yes' in result.text
    assert 'fix integrate all nve' in result.text
    assert 'run 40' in result.text
    assert result.cadence == 5
    assert result.timestep == .001
    assert result.settings['temperature_k']['source'] == 'DICE'


def _clean_sample(tmp_path, temperatures=(600.,)):
    from ase import Atoms
    from workflow_core.common.lammps_control import LammpsSamplingOptions
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from test_sampling_backend_contract import _FakeLammpsExecutor

    recipe = tmp_path/'protocol.lammps'
    recipe.write_text('units metal\natom_style atomic\ntimestep 0.001\n'
        'fix bath all nvt temp 300 300 0.1 drag 0.2\nrun 80\n')
    model = tmp_path/'model.pb'; model.write_text('fixture model, no native execution')
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (SamplingConditionGroup(
        'bulk', 'nvt', ('A',), steps_by_round=(20,), temperatures_k=temperatures,
        lammps=LammpsSamplingOptions.from_payload({})),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
        (SamplingSeedSource('A', (Atoms('H', cell=[4]*3, pbc=True),)),),
        OrderedSamplingModels((model,), ('student',)), None)
    backend = LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=('H',),
        group_recipes={'bulk':recipe})
    backend.execute(SamplingExecutionRequest(evidence))
    return backend, evidence


def test_clean_protocol_survives_sample_publication_screen_replay_and_validation(tmp_path):
    backend, evidence = _clean_sample(tmp_path)
    assert backend.validate_current(evidence).complete
    source = backend.open_current(evidence).source
    rows = source.catalog()
    assert len(rows) == 1
    frames = tuple(source.replay((rows[0].structure_id,), chunk_size=1))
    assert frames[0][0].info['md_step'] == 20
    assert 'nvt temp 600 600 0.1 drag 0.2' in (tmp_path/'sample/tasks/task_000000/input.lammps').read_text()


def test_sample_validation_reads_hardlinked_model_once_per_pass(tmp_path, monkeypatch):
    from pathlib import Path
    backend, evidence = _clean_sample(tmp_path, (300., 600., 900.))
    model = tmp_path/'model.pb'
    identity = (model.stat().st_dev, model.stat().st_ino)
    original_open = Path.open
    reads = []
    def opened(path, *args, **kwargs):
        if args and args[0] == 'rb' and path.exists():
            stat = path.stat()
            if (stat.st_dev, stat.st_ino) == identity:
                reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    assert backend.validate_current(evidence).complete
    assert len(reads) == 1, f'same immutable model read {len(reads)} times'


def test_public_loader_keeps_recipe_only_requests_unknown_without_opening_recipe(tmp_path):
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.config import load_run_config
    run = _write_lammps_run(tmp_path, sampler={'backend':'LAMMPS'}, sample={
        'seeds':{'A':{'source':'data/A.xyz'}}, 'round_count':2,
        'condition_groups':{'native':{'recipe':'recipes/not-opened.lammps'}}})
    config = load_run_config(run)
    task, = config.sampling_plan.tasks_for_round(1)
    assert task.kind == 'native'
    assert task.steps is None and task.temperature_k is None and task.pressure_bar is None
    assert task.protocol_request.ensemble is None


def test_public_nvt_without_mode_uses_field_binding_even_when_all_conditions_are_present(tmp_path):
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.config import load_run_config
    run = _write_lammps_run(tmp_path, sampler={'backend':'LAMMPS'}, sample={
        'seeds':{'A':{'source':'data/A.xyz'}}, 'round_count':1,
        'condition_groups':{'native':{'kind':'nvt', 'recipe':'recipes/native.lammps',
            'temperatures_k':[600], 'steps':20,
            'lammps':{'control':{'temperature_damping_ps':.2}}}}})
    task, = load_run_config(run).sampling_plan.tasks_for_round(0)
    assert task.protocol_request.temperature_k == 600
    assert task.protocol_request.temperature_damping_ps == .2


def _native_protocol_case(tmp_path, fields=None, minimize=False, source=None, executor_factory=None, model_suffix='.pb', seed_selections=()):
    from test_sampling_conditions import _write_lammps_run
    from test_sampling_backend_contract import _FakeLammpsExecutor
    from ase import Atoms
    from workflow_core.lifecycle.config import load_run_config
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    run = _write_lammps_run(tmp_path, sampler={'backend':'LAMMPS'}, sample={
        'seeds':{'A':{'source':'data/A.xyz'}}, 'round_count':1,
        'condition_groups':{'native':{'recipe':'recipes/native.lammps', **(fields or {})}}})
    recipe = tmp_path/'recipes/native.lammps'
    recipe.write_text('units metal\natom_style atomic\nvelocity all create 300 917\nfix integrate all nve\nrun 20\n')
    if minimize:recipe.write_text('units metal\natom_style atomic\nmin_style cg\nminimize 0 0.01 3 100\n')
    if source is not None:recipe.write_text(source)
    model = tmp_path/f'model{model_suffix}';model.write_text('parser fixture model')
    evidence = SamplingEvidenceRequest(run, 0, 0, tmp_path/'sample', load_run_config(run).sampling_plan,
        (SamplingSeedSource('A', (Atoms('H',cell=[4]*3,pbc=True),)),),
        OrderedSamplingModels((model,), ('student',)), None, seed_selections=seed_selections)
    class Executor(_FakeLammpsExecutor):
        def run(self, task_dir, task):
            super().run(task_dir, task)
            if minimize:
                trajectory=task_dir/'traj.lammpstrj'
                content=trajectory.read_text();first,second=content.split('ITEM: TIMESTEP\n20\n')
                trajectory.write_text(first)
                (task_dir/'protocol_endpoint.dump').write_text('ITEM: TIMESTEP\n3\n'+second)
            (task_dir/'native_protocol_complete.out').write_text(f'DICE_NATIVE_PROTOCOL_COMPLETE {3 if minimize else 20}\n')
    executor = Executor() if executor_factory is None else executor_factory(Executor)
    from workflow_app.composition import sampling_backend_binding
    from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory
    backend = LammpsSamplingBackendFactory(executor).bind(sampling_backend_binding(load_run_config(run)))
    backend.execute(SamplingExecutionRequest(evidence))
    return backend,evidence


def test_ordinary_pt2_protocol_enables_atom_mapping_before_loading_structure(tmp_path):
    _native_protocol_case(tmp_path, model_suffix='.pt2')
    setup = (tmp_path/'sample/tasks/task_000000/dice_task_setup.lammps').read_text()
    assert 'pair_style deepmd ' in setup
    assert setup.index('atom_modify map yes') < setup.index('read_data conf.lmp')


def test_native_protocol_publishes_resolved_multiframe_seed_selection(tmp_path):
    import json
    from workflow_backends.sampling import SamplingSeedSelection, seed_selection_manifest_payload
    selections = (SamplingSeedSelection('A', 'a'*64, 1, 2, (0,), 'b'*64, 'fixture'),)
    _native_protocol_case(tmp_path, seed_selections=selections)
    assert json.loads((tmp_path/'sample/seed_selection.json').read_text()) == seed_selection_manifest_payload(
        iter_index=0, round_index=0, selections=selections)


def test_multiframe_expansion_retains_native_protocol_scientific_request(tmp_path):
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.config import load_run_config
    run = _write_lammps_run(tmp_path, sampler={'backend':'LAMMPS'}, sample={
        'seeds':{'A':{'source':'data/A.xyz','max_frames':2}}, 'round_count':1,
        'condition_groups':{'native':{'recipe':'recipes/nvt.lammps','temperatures_k':[600],
            'steps':20,'targets':{'thermostat':'bath'}}}})
    plan = load_run_config(run).sampling_plan
    original, = plan.tasks_for_round(0)
    tasks = plan.expand_source_frames({'A':(0,2)}).tasks_for_round(0)
    assert len(tasks) == 2
    assert all(task.protocol_request == original.protocol_request for task in tasks)
    assert [task.source_frame_index for task in tasks] == [0,2]


def test_native_protocol_stages_declared_auxiliary_files_and_validates_them(tmp_path):
    (tmp_path/'temperature.txt').write_text('600\n')
    backend,evidence = _native_protocol_case(tmp_path,
        fields={'inputs':{'fields/temperature.txt':'temperature.txt'}},
        source='units metal\natom_style atomic\nvariable target file fields/temperature.txt\n'
               'fix bath all nvt temp ${target} ${target} .1\nrun 20\n')
    staged = tmp_path/'sample/tasks/task_000000/fields/temperature.txt'
    assert staged.read_text() == '600\n'
    assert backend.validate_current(evidence).complete
    staged.write_text('900\n')
    assert not backend.validate_current(evidence).complete


def test_native_md_failure_retains_valid_prefix_and_drains_other_tasks(tmp_path):
    import json
    def executor_factory(base):
        class Executor(base):
            calls = []
            def run(self, root, task):
                self.calls.append(root.name)
                super().run(root, task)
                if root.name == 'task_000000':
                    (root/'native_protocol_complete.out').unlink()
                    with (root/'traj.lammpstrj').open('a') as f:
                        f.write('ITEM: TIMESTEP\n30\nITEM: NUMBER OF ATOMS\n')
                    raise RuntimeError('native task failed after a complete frame')
        return Executor()
    backend,evidence = _native_protocol_case(tmp_path, fields={'temperatures_k':[300,600]},
        source='units metal\nfix bath all nvt temp 900 900 .1\nrun 20\n', executor_factory=executor_factory)
    assert backend.executor.calls == ['task_000000','task_000001']
    assert backend.validate_current(evidence).complete
    source = backend.open_current(evidence).source
    assert source.failed_task_count == 1
    assert len(source.catalog()) == 2
    result = json.loads((tmp_path/'sample/tasks/task_000000/protocol_result.json').read_text())
    assert result['status'] == 'failed'


def test_recipe_only_nve_publishes_replayable_native_result_without_fake_temperature(tmp_path):
    backend,evidence=_native_protocol_case(tmp_path)
    assert backend.validate_current(evidence).complete
    source = backend.open_current(evidence).source
    row, = source.catalog()
    assert row.md_step == 20
    context = __import__('json').loads((tmp_path/'sample/tasks/task_000000/task_context.json').read_text())
    assert context['effective']['temperature_k'] is None
    assert context['request']['temperature_k'] is None
    frame, = next(source.replay((row.structure_id,), chunk_size=1))
    assert frame.get_chemical_symbols() == ['H']


def test_native_result_requires_its_original_terminal_marker(tmp_path):
    backend,evidence=_native_protocol_case(tmp_path)
    (tmp_path/'sample/tasks/task_000000/native_protocol_complete.out').unlink()
    assert not backend.validate_current(evidence).complete


def test_native_protocol_rejects_shortened_explicit_budget(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='budget'):
        _native_protocol_case(tmp_path, {'steps':40})


def test_native_minimize_exposes_early_terminal_even_below_dump_cadence(tmp_path):
    backend,evidence=_native_protocol_case(tmp_path,minimize=True)
    source=backend.open_current(evidence).source
    rows=source.catalog()
    assert [r.optimizer_step for r in rows]==[0,3]
    assert [r.md_step for r in rows]==[None,None]
    frames=[frame for chunk in source.replay(tuple(r.structure_id for r in rows),chunk_size=2) for frame in chunk]
    assert frames[-1].positions[0,0]==.1


def test_native_box_conversion_precedes_automatic_observation_dump(tmp_path):
    _native_protocol_case(tmp_path,source='units metal\natom_style atomic\n'
        'change_box all triclinic\nfix bath all npt temp 300 300 .1 tri 0 0 1\nrun 20\n')
    root=tmp_path/'sample/tasks/task_000000'
    text=(root/'input.lammps').read_text().replace('include dice_task_setup.lammps',
        (root/'dice_task_setup.lammps').read_text())
    assert text.index('change_box all triclinic') < text.index('dump dice_observations ')
    assert text.index('dump dice_observations ') < text.index('run 20')


def test_native_dump_validation_binds_initial_types_and_positive_cell(tmp_path):
    import pytest
    from ase import Atoms
    from workflow_backends.lammps.protocol_result import scan_protocol_dump
    backend,evidence=_native_protocol_case(tmp_path)
    path=tmp_path/'sample/tasks/task_000000/traj.lammpstrj'
    original=path.read_text()
    path.write_text(original.replace('\n1 1 ', '\n1 2 '))
    with pytest.raises(ValueError,match='identity'):
        scan_protocol_dump(path,type_map=('H','He'),seed=Atoms('H'))
    path.write_text(original.replace('0 4\n0 4\n0 4','4 0\n0 4\n0 4'))
    with pytest.raises(ValueError,match='cell'):
        scan_protocol_dump(path,type_map=('H',),seed=Atoms('H'))


def test_full_native_model_binding_preserves_type_number_meanings():
    import pytest
    from workflow_backends.lammps.protocol_sampler import resolve_native_type_map
    source='read_data original.data\npair_style deepmd old.pb\npair_coeff * * O Zr Y\ngroup oxygen type 1\n'
    assert resolve_native_type_map(source, ('Zr','Y','O')) == ('O','Zr','Y')
    with pytest.raises(ValueError,match='element map'):
        resolve_native_type_map(source, ('H',))


def test_preflight_distinguishes_different_ase_algorithms_on_same_endpoints(tmp_path):
    from ase import Atoms
    from ase.io import write
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.launch import prepare_execution_suffix, prepare_lifecycle_launch
    groups={method:{'kind':'neb','images':2,'fmax_ev_a':.1,'max_steps':5,'method':method}
            for method in ('neb','dyneb')}
    run=_write_lammps_run(tmp_path,sampler={'backend':'LAMMPS'},sample={
        'round_count':1,'seeds':{'A':{'source':'data/A.xyz'}},'condition_groups':groups})
    initial=Atoms('H',cell=[4]*3,pbc=True);final=initial.copy();final.positions[0,0]=.1
    write(tmp_path/'data/A.xyz',[initial,final],format='extxyz')
    prepare_execution_suffix(prepare_lifecycle_launch(run),first_block='r0/sample',require_environment=False)


def test_declarations_can_target_different_baths_in_the_same_native_recipe(tmp_path):
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.config import load_run_config
    run=_write_lammps_run(tmp_path,sampler={'backend':'LAMMPS'},sample={
        'round_count':1,'seeds':{'A':{'source':'data/A.xyz'}},
        'condition_groups':{target:{'recipe':'recipes/native.lammps',
            'temperatures_k':[600],'targets':{'thermostat':target}} for target in ('hot','cold')}})
    tasks=load_run_config(run).sampling_plan.tasks_for_round(0)
    assert [task.protocol_request.thermostat for task in tasks]==['hot','cold']


def test_clean_protocols_preserve_allocated_mps_slots_and_global_task_order(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from threading import Barrier, Lock
    from workflow_backends.execution import current_operation_environment_overrides
    from workflow_runtime.lifecycle.environment import OperationContext
    from workflow_runtime.execution.allocation import Allocation
    barrier,lock=Barrier(4,timeout=3),Lock()
    assignments,sessions={},[]
    def factory(base):
        class Executor(base):
            context=OperationContext(role='sampler.lammps.tf',profile_id='test',source_scripts=(),
                executables={},runtime={},allocation=Allocation(1,2,4,2048,'direct','pool',2))
            @contextmanager
            def open_mps_device_sessions(self,root,devices):
                sessions.append(('start',devices))
                try:yield {d:{'CUDA_MPS_PIPE_DIRECTORY':'/fixture/'+d} for d in devices}
                finally:sessions.append(('stop',devices))
            def run(self,root,task):
                env=current_operation_environment_overrides()
                with lock:assignments[root.name]=env.copy()
                barrier.wait()
                super().run(root,task)
        return Executor()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    backend,evidence=_native_protocol_case(tmp_path,fields={'temperatures_k':[300,600,900,1200]},
        source='units metal\natom_style atomic\nfix bath all nvt temp 300 300 .1\nrun 20\n',executor_factory=factory)
    assert len(assignments)==4
    assert sorted(env['CUDA_VISIBLE_DEVICES'] for env in assignments.values())==['0','0','1','1']
    assert sessions==[('start',('0','1')),('stop',('0','1'))]
    assert [row.task_index for row in backend.open_current(evidence).source.catalog()]==[0,1,2,3]


def test_clean_protocols_honor_existing_lammps_parallel_task_policy(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from threading import Barrier
    from workflow_runtime.lifecycle.environment import OperationContext,LammpsExecutionPolicy
    from workflow_runtime.execution.allocation import Allocation
    barrier,sessions=Barrier(2,timeout=1),[]
    def factory(base):
        class Executor(base):
            context=OperationContext(role='sampler.lammps.tf',profile_id='test',source_scripts=(),
                executables={},runtime={},allocation=Allocation(1,1,4,2048,'direct','pool',1))
            execution_policy=LammpsExecutionPolicy(2,50)
            @contextmanager
            def open_mps_session(self,root):
                sessions.append('start')
                try:yield {'CUDA_MPS_PIPE_DIRECTORY':'/fixture/0'}
                finally:sessions.append('stop')
            def run(self,root,task):
                barrier.wait();super().run(root,task)
        return Executor()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    _native_protocol_case(tmp_path,fields={'temperatures_k':[300,600]},
        source='units metal\natom_style atomic\nfix bath all nvt temp 300 300 .1\nrun 20\n',executor_factory=factory)
    assert sessions==['start','stop']


def test_recipe_only_preflight_reports_inherited_settings_before_native_work(tmp_path):
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch
    from workflow_app.composition import preflight_sampling_protocols
    run = _write_lammps_run(tmp_path, sampler={'backend':'LAMMPS'}, sample={
        'seeds':{'A':{'source':'data/A.xyz'}}, 'round_count':1,
        'condition_groups':{'native':{'recipe':'recipes/native.lammps'}}})
    (tmp_path/'recipes/native.lammps').write_text('units metal\natom_style atomic\n'
        'fix bath all nvt temp 450 450 0.15\nrun 40\n')
    summary, = preflight_sampling_protocols(prepare_lifecycle_launch(run), first_block='r0/sample')
    assert summary['settings']['temperature_k']['value'] == 450
    assert summary['settings']['temperature_k']['source'] == 'native'
    assert not (tmp_path/'iter0').exists()


def test_core_protocol_intent_carries_only_requested_conditions():
    from workflow_core.lifecycle.sampling import SamplingTaskIntent
    from workflow_core.common.native_protocol import NativeProtocolRequest
    task = SamplingTaskIntent('task_000000','A','native',None,
        kind='native',condition_group_id='native',sampling_role_id='A',selection_group_id='joint',
        protocol_request=NativeProtocolRequest())
    assert task.steps is None and task.temperature_k is None


def test_named_target_and_timestep_override_leave_other_bath_and_velocity_untouched():
    from workflow_core.common.native_protocol import NativeProtocolRequest
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol
    source=('units metal\ntimestep 0.001\nvelocity hot create 400 91\n'
            'fix cold cold nvt temp 300 300 .2\nfix hot hot langevin 900 900 .1 123\n'
            'fix integrate hot nve\nrun 40\n')
    result=compile_lammps_protocol(source,NativeProtocolRequest(temperature_k=700,
        thermostat='hot',timestep_ps=.002))
    assert 'fix cold cold nvt temp 300 300 .2' in result.text
    assert 'fix hot hot langevin 700 700 .1 123' in result.text
    assert 'velocity hot create 400 91' in result.text
    assert 'timestep 0.002' in result.text
    assert result.timestep_ps == .002


def test_program_preflight_checks_native_controller_before_any_mutation(tmp_path):
    import pytest
    from test_sampling_conditions import _write_lammps_run
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch
    from workflow_app.composition import preflight_sampling_protocols
    run=_write_lammps_run(tmp_path,sampler={'backend':'LAMMPS'},sample={
        'seeds':{'A':{'source':'data/A.xyz'}},'round_count':1,
        'condition_groups':{'program':{'kind':'nvt_program','recipe':'recipes/native.lammps',
            'segments':[{'temperature_k':300,'steps':20}]}}})
    (tmp_path/'recipes/native.lammps').write_text('units metal\natom_style atomic\nrun 20\n')
    with pytest.raises(ValueError,match='thermostat'):
        preflight_sampling_protocols(prepare_lifecycle_launch(run),first_block='r0/sample')
    assert not (tmp_path/'iter0').exists()
