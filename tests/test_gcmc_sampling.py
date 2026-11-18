from __future__ import annotations

import json
import numpy as np
import pytest

from workflow_core.lifecycle.config import load_run_config
from test_sampling_conditions import _write_lammps_run
from test_gcmc_phase import official, Zero, inputs


def test_deferred_nep_gcmc_fails_at_public_config_boundary(tmp_path):
    from test_sampling_conditions import _write_run
    run = _write_run(tmp_path, sample={
        'round_count': 1, 'seeds': {'A': {'source': 'data/A.xyz'}},
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303], 'cycles': 1,
            'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.1], 'moves': 10, 'eos': {'kind': 'ideal'}}}}})
    payload = json.loads(run.read_text())
    payload['sampler'] = {'backend': 'GPUMD'}
    run.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='GPUMD explicit sampling supports only neb, relax, nvt, and npt'):
        load_run_config(run)


@pytest.mark.parametrize('ensemble', ['nvt', 'nve', 'npt'])
def test_hybrid_md_intent_keeps_native_recipe_and_reservoir_pressure_separate(tmp_path, ensemble):
    md = {'ensemble': ensemble, 'recipe': 'recipes/hybrid.lammps', 'steps': 7, 'every': 3}
    if ensemble == 'npt':
        md['pressure_bar'] = 1.5
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz'}}, 'round_count': 1,
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303],
            'cycles': 2, 'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.1],
                              'moves': 137, 'eos': {'kind': 'ideal'}}, 'md': md}},
    })
    config = load_run_config(run)
    task = config.sampling_plan.tasks[0]
    assert task.recipe_binding == 'ads'
    assert task.pressure_bar == .1
    assert task.gcmc.md.ensemble == ensemble
    assert (task.gcmc.md.steps, task.gcmc.md.every) == (7, 3)
    assert task.gcmc.md.pressure_bar == (1.5 if ensemble == 'npt' else None)
    from workflow_app.composition import sampling_backend_binding
    binding = sampling_backend_binding(config)
    assert binding.recipes[0].path == tmp_path/'recipes/hybrid.lammps'
    assert binding.guest_inputs['ads'] == tmp_path/'data/guest.xyz'
    from types import SimpleNamespace
    from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory
    backend = LammpsSamplingBackendFactory(SimpleNamespace(context=None)).bind(binding)
    assert backend.md_recipes == {'ads': tmp_path/'recipes/hybrid.lammps'}
    invalid = json.loads(run.read_text())
    invalid['sample']['condition_groups']['ads']['md']['kwargs'] = {}
    run.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='kwargs'):
        load_run_config(run)


def test_public_pure_mc_config_expands_pressure_temperature_and_round_budget(tmp_path):
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz', 'selection_group': 'joint'},
                  'B': {'source': 'data/B.xyz', 'selection_group': 'joint'}},
        'round_count': 2,
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303, 323],
            'cycles': [2, 3], 'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.01, .1],
                                  'moves': 137, 'eos': {'kind': 'ideal'}}}},
    })
    config = load_run_config(run)
    tasks = config.sampling_plan.tasks_for_round(1)
    assert len(tasks) == 8
    assert [(t.sampling_role_id, t.temperature_k, t.pressure_bar) for t in tasks] == [
        (role, t, p) for role in ('A', 'B') for t in (303., 323.) for p in (.01, .1)]
    assert all(t.selection_group_id == 'joint' and t.cycles == 3 for t in tasks)
    assert all(t.steps is None and t.gcmc.moves == 137 and t.gcmc.every == 100 for t in tasks)
    assert tasks[0].gcmc.translate_a == .75
    assert tasks[0].gcmc.de_max_ev is None
    from workflow_core.lifecycle.sampling import SamplingTaskIntent
    t = tasks[0]
    intent = SamplingTaskIntent(t.task_id, t.system_id, t.ensemble, t.temperature_k, t.pressure_bar,
                                condition_group_id=t.condition_group_id, sampling_role_id=t.sampling_role_id,
                                selection_group_id=t.selection_group_id, kind=t.kind,
                                gcmc=t.gcmc, cycles=t.cycles)
    assert intent.gcmc == t.gcmc and intent.cycles == 3
    assert intent.ensemble == 'gcmc_md'
    from workflow_app.composition import sampling_backend_binding
    binding = sampling_backend_binding(config)
    assert binding.guest_inputs == {'ads': tmp_path/'data/guest.xyz'}
    assert binding.recipes == ()


def test_public_mc_advanced_controls_are_resolved_without_passthrough(tmp_path):
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz'}}, 'round_count': 1,
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303],
            'cycles': 2, 'random_seed': 42, 'limits': {'max_guests': 12},
            'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.1], 'moves': 137,
                   'every': 20, 'translate_a': .4, 'rotate_deg': 45,
                   'eos': {'kind': 'fixed_phi', 'phi': .9}, 'de_max_ev': 5.,
                   'weights': dict(insertion=.5, deletion=.5, translation=0., rotation=0., reinsertion=0.),
                   'overlap': {'scale': .5, 'radii_a': {'H': 1.2}}}}},
    })
    p = load_run_config(run).sampling_plan.tasks[0].gcmc
    assert (p.phi, p.translate_a, p.rotate_deg, p.every) == (.9, .4, 45., 20)
    assert (p.max_guests, p.random_seed, p.de_max_ev) == (12, 42, 5.)
    assert p.radii_a == (('H', 1.2),)
    original = json.loads(run.read_text())
    for section, key, value in [('mc', 'de_max_ev', -1), ('mc', 'moves', True),
                                ('mc', 'pressures_bar', [0]), ('mc', 'flames_kwargs', {}),
                                ('limits', 'unknown', 1)]:
        invalid = json.loads(json.dumps(original))
        invalid['sample']['condition_groups']['ads'][section][key] = value
        run.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=key):
            load_run_config(run)


def test_pr_config_and_actual_flames_fugacity_agree(tmp_path, official):
    from workflow_backends.gcmc.eos import reservoir_phi
    from workflow_backends.gcmc.phase import McPhase
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz'}}, 'round_count': 1,
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303],
            'cycles': 1, 'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.1, 1.],
                'moves': 1, 'eos': {'kind': 'pr', 'tc_k': 304.1282, 'pc_bar': 73.773, 'omega': .22394}}}}})
    parameters = load_run_config(run).sampling_plan.tasks[0].gcmc
    host, guest, _ = inputs()
    values = []
    for pressure in (.1, 1.):
        phi = reservoir_phi(parameters, temperature_k=303., pressure_bar=pressure, guest=guest)
        values.append(phi)
        with McPhase(official, host, guest, Zero(), temperature_k=303., pressure_bar=pressure,
                     random_seed=1, velocity_seed=2, output_dir=tmp_path/f'p{pressure}', device='cpu', phi=phi) as phase:
            controls = phase.evidence()['controls']
            assert controls['fugacity_pa'] == pytest.approx(phi*pressure*1e5)
            assert controls['phi'] == phi
    assert .999 < values[0] < 1.
    assert .994 < values[1] < .9955
    from dataclasses import replace
    with pytest.raises(ValueError, match='tc_k'):
        replace(parameters, tc_k=None)
    with pytest.raises(ValueError, match='PR|pr'):
        replace(parameters, eos_kind='ideal')


@pytest.mark.parametrize('member_count', [1, 2])
def test_mc_sampling_capability_publishes_variable_n_and_replays_without_checkpoints(tmp_path, official, member_count):
    from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures

    host, guest, pressure = inputs()
    host.info.update(student_uncertainty=99., force_uncertainty=99., force_std_max=99.)
    guest_path = tmp_path/'guest.extxyz'
    write_extxyz_structures(guest_path, [guest])
    model = tmp_path/'model.fixture'
    model.write_text('analytic-evaluator-fixture')
    models = [model]
    if member_count == 2:
        second = tmp_path/'second.fixture'
        second.write_text('second-member-fixture')
        models.append(second)
    class Offset(Zero):
        def calculate(self, atoms=None, properties=('energy',), system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results['forces'][:] = .2
            self.results['energy'] = -.2*float(atoms.positions.sum())
    class Provider:
        def preflight(self, paths):
            assert paths == tuple(models)
        def create(self, path):
            return Zero() if path == model else Offset()
    plan = ExplicitSamplingPlan(
        (SeedRole('A', 'joint'),),
        (SamplingConditionGroup('ads', 'gcmc_md', ('A',), temperatures_k=(303.,),
                               pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                                   (2,), moves=137, every=20, random_seed=139)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
                                      (SamplingSeedSource('A', (host,)),),
                                      OrderedSamplingModels(tuple(models), tuple(f'member-{i}' for i in range(member_count))), None)
    backend = GcmcSamplingNativeOps(calculator_provider=Provider(), guest_sources={'ads': guest_path},
                                   gcmc_class=official, device='cpu')
    import fcntl
    evidence.sample_dir.mkdir()
    sentinel = evidence.sample_dir/'tasks'/'live'
    sentinel.parent.mkdir()
    sentinel.write_text('owned by another sample writer')
    with (evidence.sample_dir/'.gcmc-stage.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active.*writer'):
            backend.execute(SamplingExecutionRequest(evidence))
        assert sentinel.read_text() == 'owned by another sample writer'
        assert not tuple(evidence.sample_dir.glob('.previous-sample-*'))
    result = backend.execute(SamplingExecutionRequest(evidence))
    assert result.task_count == 1
    assert backend.validate_current(evidence).complete
    source = backend.open_current(evidence).source
    records = source.catalog()
    assert len(records) == 13
    assert [r.mc_attempts for r in records] == list(range(20, 275, 20))
    assert all(r.phase == 'gcmc' and r.md_time_fs == 0. for r in records)
    assert all(r.md_step == 0 for r in records)
    if member_count == 1:
        assert all(r.student_uncertainty is None for r in records)
    else:
        assert all(r.student_uncertainty == pytest.approx(np.sqrt(3)*.1) for r in records)
    frames = [a for chunk in source.replay(tuple(r.structure_id for r in records), chunk_size=3) for a in chunk]
    assert [a.info['mc_attempts'] for a in frames] == list(range(20, 275, 20))
    assert len(set(len(a) for a in frames)) > 1
    assert all(a.info['selection_group_id'] == 'joint' for a in frames)
    assert all('dice_force_member_000' in a.arrays for a in frames)
    if member_count == 1:
        assert all(not any(k in a.info for k in ('student_uncertainty', 'force_uncertainty', 'force_std_max')) for a in frames)
    task_root = tmp_path/'sample/tasks/task_000000'
    # The block-level compression validator expects ordinary task-local graph
    # presentations, even when FLAMES drives MC and MD has nested directories.
    for index, path in enumerate(models):
        assert (task_root/f'graph_{index:03d}{path.suffix}').read_bytes() == path.read_bytes()
    archive = tmp_path/'operational-evidence'
    archive.mkdir()
    for path in tuple(task_root.iterdir()):
        if path.name not in {'gcmc_result.json', 'observations.extxyz', 'final_structure.extxyz'}:
            path.rename(archive/path.name)
    assert backend.validate_current(evidence).complete
    from workflow_backends.sample_recovery import validate_sample_artifacts
    from workflow_core.lifecycle.sampling import SampleRecoveryFacts, SamplingTaskIntent
    t = plan.tasks_for_round(0)[0]
    intent = SamplingTaskIntent(t.task_id, t.system_id, t.ensemble, t.temperature_k, t.pressure_bar,
                               condition_group_id=t.condition_group_id, sampling_role_id=t.sampling_role_id,
                               selection_group_id=t.selection_group_id, kind=t.kind, gcmc=t.gcmc, cycles=t.cycles)
    facts = SampleRecoveryFacts('LAMMPS', 'pt', evidence.models.member_ids, ('He', 'Ne'),
                                ('A',), (intent,), None, False, False)
    assert validate_sample_artifacts(evidence.sample_dir, facts)
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from test_sampling_backend_contract import _IdentitySamplingModelProvider
    coordinator = LifecycleSamplingCoordinator(backend, _IdentitySamplingModelProvider())
    assert coordinator.validate_completion(evidence).complete
    assert coordinator.open_observations(evidence).source.catalog() == records
    if member_count == 1:
        # Even a resealed file must satisfy the intrinsic acquisition contract.
        import hashlib
        from ase.io import read, write
        result_path = task_root/'gcmc_result.json'
        manifest_bytes = result_path.read_bytes()
        for name in ('observations.extxyz', 'final_structure.extxyz'):
            path = task_root/name
            original_bytes = path.read_bytes()
            for corruption in ('duplicate_id', 'negative_membership', 'missing_velocity', 'singular_cell'):
                structures = read(path, index=':')
                frame = structures[0]
                if corruption == 'duplicate_id':
                    frame.arrays['dice_atom_id'][:] = 0
                elif corruption == 'negative_membership':
                    frame.arrays['dice_molecule_id'][0] = -1
                elif corruption == 'missing_velocity':
                    frame.set_array('momenta', None)
                else:
                    frame.set_cell(np.zeros((3, 3)))
                write(path, structures, format='extxyz')
                manifest = json.loads(manifest_bytes)
                manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                result_path.write_text(json.dumps(manifest))
                assert not backend.validate_current(evidence).complete, (name, corruption)
                path.write_bytes(original_bytes)
                result_path.write_bytes(manifest_bytes)
    if member_count == 2:
        from dataclasses import replace
        changed_members = replace(evidence, models=OrderedSamplingModels(tuple(models), ('member-1', 'member-0')))
        assert not backend.validate_current(changed_members).complete
        result_path = task_root/'gcmc_result.json'
        manifest = json.loads(result_path.read_text())
        manifest['observations'][0]['force_deviation'] = 0.
        result_path.write_text(json.dumps(manifest))
        assert not backend.validate_current(evidence).complete


@pytest.mark.parametrize('fault', ['before_mc', 'inference', 'before_validation', 'after_validation', 'before_publish', 'after_publish'])
def test_interrupted_mc_sample_restarts_without_duplicate_candidates(tmp_path, official, monkeypatch, fault):
    from dataclasses import replace
    import numpy as np
    from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures

    host, guest, pressure = inputs()
    guest_path, model = tmp_path/'guest.extxyz', tmp_path/'model.fixture'
    write_extxyz_structures(guest_path, [guest])
    model.write_text('analytic-evaluator-fixture')

    class Interruptible(Zero):
        def get_property(self, name, atoms=None, allow_calculation=True):
            if fault == 'inference' and provider.interrupt and name == 'forces' and atoms.info.get('mc_attempts', 0) >= 160:
                raise KeyboardInterrupt('injected inference interruption')
            return super().get_property(name, atoms, allow_calculation)

    class Provider:
        interrupt = False
        def preflight(self, paths):
            assert paths == (model,)
        def create(self, path):
            if fault == 'before_mc' and self.interrupt:
                raise KeyboardInterrupt('injected pre-MC interruption')
            return Interruptible()

    provider = Provider()
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (
        SamplingConditionGroup('ads', 'gcmc_md', ('A',), temperatures_k=(303.,),
                               pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                                   (3,), moves=137, every=20, random_seed=139)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'reference', plan,
                                      (SamplingSeedSource('A', (host,)),),
                                      OrderedSamplingModels((model,), ('member-0',)), None)
    backend = GcmcSamplingNativeOps(calculator_provider=provider, guest_sources={'ads': guest_path},
                                   gcmc_class=official, device='cpu')
    backend.execute(SamplingExecutionRequest(evidence))
    reference = backend.open_current(evidence).source
    interrupted = replace(evidence, sample_dir=tmp_path/'interrupted')
    from workflow_backends.gcmc import adapter
    original_validate = adapter.validate_task
    def interrupt_validation(path, **kwargs):
        selected = provider.interrupt and path.name.startswith('.task_000000.gcmc-')
        if selected and fault == 'before_validation':
            raise KeyboardInterrupt('injected endpoint validation interruption')
        result = original_validate(path, **kwargs)
        if selected and fault == 'after_validation':
            raise KeyboardInterrupt('injected endpoint validation interruption')
        return result
    monkeypatch.setattr(adapter, 'validate_task', interrupt_validation)
    import os
    original_replace = os.replace
    def interrupt_publication(source, destination):
        target = interrupted.sample_dir/'tasks/task_000000'
        if provider.interrupt and destination == target and fault == 'before_publish':
            raise KeyboardInterrupt('injected publication interruption')
        result = original_replace(source, destination)
        if provider.interrupt and destination == target and fault == 'after_publish':
            raise KeyboardInterrupt('injected publication interruption')
        return result
    monkeypatch.setattr(os, 'replace', interrupt_publication)
    provider.interrupt = True
    with pytest.raises(KeyboardInterrupt, match='interruption'):
        backend.execute(SamplingExecutionRequest(interrupted))
    assert backend.validate_current(interrupted).complete == (fault == 'after_publish')
    provider.interrupt = False
    backend.execute(SamplingExecutionRequest(interrupted))
    recovered = backend.open_current(interrupted).source
    assert recovered.catalog() == reference.catalog()
    ids = tuple(row.structure_id for row in reference.catalog())
    for left, right in zip(reference.replay(ids, chunk_size=4), recovered.replay(ids, chunk_size=4), strict=True):
        for a, b in zip(left, right, strict=True):
            for name in a.arrays:
                np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
    # Re-executing an already consumer-runnable result does not duplicate work.
    backend.execute(SamplingExecutionRequest(interrupted))
    assert backend.open_current(interrupted).source.catalog() == recovered.catalog()


@pytest.mark.parametrize('fault', ['interrupt', 'after_md', 'task_failure', 'task_failure_truncated', 'task_failure_md_prefix'])
def test_hybrid_acquisition_restarts_failed_sample_without_duplicate_observations(tmp_path, official, fault):
    from dataclasses import replace
    from types import SimpleNamespace
    from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters, GcmcMdParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    host, guest, pressure = inputs()
    guest_path, model = tmp_path/'guest.extxyz', tmp_path/'model.fixture'
    write_extxyz_structures(guest_path, [guest])
    model.write_text('fixture')
    class Provider:
        def preflight(self, paths): pass
        def create(self, path): return Zero()
    class NativeBoundary:
        identity = {'backend': 'analytic-fixed-N-test'}
        fail = False
        calls = []
        def run(self, atoms, *, parameters, temperature_k, start_step, output_dir):
            self.calls.append(start_step)
            if self.fail and len(self.calls) == 1 and fault == 'task_failure_md_prefix':
                from workflow_backends.gcmc.lammps import MdSegmentFailure
                class Partial(MdSegmentFailure):
                    @property
                    def observations(inner):
                        for step in (3, 6):
                            frame = atoms.copy()
                            frame.info.update(md_steps=step, phase_step=step)
                            yield frame
                        raise ValueError('truncated native tail')
                raise Partial('failed native MD', SimpleNamespace(timestep_fs=.5))
            if self.fail and len(self.calls) == 3 and fault not in ('after_md', 'task_failure_md_prefix'):
                if fault == 'interrupt':
                    raise KeyboardInterrupt('interrupted native MD')
                if fault == 'task_failure_truncated':
                    with (output_dir.parent/'observations.extxyz').open('a') as stream:
                        stream.write('2\ntruncated frame\nH 0 0 0\n')
                raise RuntimeError('failed native MD')
            endpoint = atoms.copy()
            endpoint.positions += .001
            observations = []
            for step in range(start_step+1, start_step+parameters.steps+1):
                if step % parameters.every == 0:
                    frame = endpoint.copy()
                    frame.info.update(md_steps=step, phase_step=step-start_step)
                    observations.append(frame)
            class Segment:
                @property
                def endpoint(inner):
                    if self.fail and len(self.calls) == 3 and fault == 'after_md':
                        raise KeyboardInterrupt('interrupted native MD endpoint')
                    return endpoint
            result = Segment()
            result.observations = observations
            result.thermo = {}
            result.prepared = SimpleNamespace(timestep_fs=.5, controls={'ensemble': 'nve'})
            return result
    runner = NativeBoundary()
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'), SeedRole('B', 'joint')), (
        SamplingConditionGroup('ads', 'gcmc_md', ('A', 'B'), temperatures_k=(303.,),
            pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                (2,), moves=137, every=100, md=GcmcMdParameters('nve', 7, 3))),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'reference', plan,
        (SamplingSeedSource('A', (host,)), SamplingSeedSource('B', (host,))),
        OrderedSamplingModels((model,), ('member-0',)), None)
    backend = GcmcSamplingNativeOps(calculator_provider=Provider(), guest_sources={'ads': guest_path},
                                   gcmc_class=official, device='cpu', md_runners={'ads': runner})
    backend.execute(SamplingExecutionRequest(evidence))
    reference = backend.open_current(evidence).source
    rows = reference.catalog()
    assert [r.phase for r in rows] == ['gcmc', 'md', 'md', 'gcmc', 'md', 'md']*2
    assert [r.mc_attempts for r in rows] == [100, 137, 137, 200, 274, 274]*2
    assert [r.md_step for r in rows] == [0, 3, 6, 7, 9, 12]*2
    interrupted = replace(evidence, sample_dir=tmp_path/'interrupted')
    runner.calls = []
    runner.fail = True
    if fault.startswith('task_failure'):
        backend.execute(SamplingExecutionRequest(interrupted))
        assert backend.validate_current(interrupted).complete
        source = backend.open_current(interrupted).source
        if fault == 'task_failure_md_prefix':
            assert len(source.catalog()) == 9  # Failed A prefix plus successful B.
            assert [r.phase for r in source.catalog()[:3]] == ['gcmc', 'md', 'md']
            assert [r.md_time_fs for r in source.catalog()[:3]] == [0., 1.5, 3.]
            manifest = json.loads((interrupted.sample_dir/'tasks/task_000000/gcmc_result.json').read_text())
            assert manifest['task_status'] == 'FAILED'
            assert manifest['retained_observations'] == 3
            assert 'truncated native tail' in manifest['failure']['message']
            assert runner.calls == [0, 0, 7]  # B still runs both cycles.
            frames = [a for chunk in source.replay(tuple(r.structure_id for r in source.catalog()), chunk_size=2) for a in chunk]
            assert len(frames) == 9
            assert all('dice_force_member_000' in a.arrays for a in frames)
            from test_gcmc_saved_screen import assert_saved_candidates_reach_screen
            kept = assert_saved_candidates_reach_screen(interrupted.sample_dir, tmp_path, keep=9)
            assert {r.structure_id for r in source.catalog()[1:3]} <= kept
            return
        assert len(source.catalog()) == 7  # A's six frames plus B's saved MC prefix.
        manifest = json.loads((interrupted.sample_dir/'tasks/task_000001/gcmc_result.json').read_text())
        assert manifest['task_status'] == 'FAILED'
        assert manifest['failure']['type'] == 'RuntimeError'
        assert manifest['retained_observations'] == 1
        assert runner.calls == [0, 7, 0]
        return
    with pytest.raises(KeyboardInterrupt, match='interrupted native'):
        backend.execute(SamplingExecutionRequest(interrupted))
    assert not backend.validate_current(interrupted).complete
    runner.fail = False
    backend.execute(SamplingExecutionRequest(interrupted))
    # A had completed before B failed. Restart repeats both A and B, not only B.
    assert runner.calls == [0, 7, 0, 0, 7, 0, 7]
    recovered = backend.open_current(interrupted).source
    assert recovered.catalog() == rows
    ids = tuple(r.structure_id for r in rows)
    for left, right in zip(reference.replay(ids, chunk_size=2), recovered.replay(ids, chunk_size=2), strict=True):
        for a, b in zip(left, right, strict=True):
            for key in a.arrays:
                np.testing.assert_array_equal(a.arrays[key], b.arrays[key])


@pytest.mark.parametrize('stress', ['missing', 'nonfinite', 'shape'])
def test_npt_missing_stress_fails_before_any_mc_or_md(tmp_path, stress):
    from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps, validate_task
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters, GcmcMdParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    host, guest, pressure = inputs()
    guest_path, model = tmp_path/'guest.extxyz', tmp_path/'model.fixture'
    write_extxyz_structures(guest_path, [guest])
    model.write_text('analytic fixture without stress')
    class InvalidStress(Zero):
        implemented_properties = ['energy', 'forces', 'stress']
        def calculate(self, atoms=None, properties=('energy',), system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results['stress'] = np.full(6, np.nan) if stress == 'nonfinite' else np.zeros(2)
    class Provider:
        def preflight(self, paths): pass
        def create(self, path): return Zero() if stress == 'missing' else InvalidStress()
    entered = []
    def forbidden_mc(*args, **kwargs):
        entered.append('mc')
        raise RuntimeError('unexpected MC startup')
    class NativeBoundary:
        def run(self, *args, **kwargs):
            entered.append('md')
            raise RuntimeError('unexpected MD startup')
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (
        SamplingConditionGroup('ads', 'gcmc_md', ('A',), temperatures_k=(303.,),
            pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                (1,), moves=2, every=1, md=GcmcMdParameters('npt', 2, 1, 1.))),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
        (SamplingSeedSource('A', (host,)),), OrderedSamplingModels((model,), ('student',)), None)
    backend = GcmcSamplingNativeOps(calculator_provider=Provider(), guest_sources={'ads': guest_path},
        gcmc_class=forbidden_mc, device='cpu', md_runners={'ads': NativeBoundary()})
    with pytest.warns(RuntimeWarning, match='NPT.*stress'):
        outcome = backend.execute(SamplingExecutionRequest(evidence))
    assert outcome.failed_task_count == 1
    assert entered == []
    result = validate_task(evidence.sample_dir/'tasks/task_000000')
    assert result['task_status'] == 'FAILED' and result['retained_observations'] == 0


def test_empty_mc_catalog_is_complete_and_damaged_inventory_is_not(tmp_path, official):
    from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures

    host, guest, pressure = inputs()
    guest_path, model = tmp_path/'guest.extxyz', tmp_path/'model.fixture'
    write_extxyz_structures(guest_path, [guest])
    model.write_text('fixture')
    class Provider:
        def preflight(self, paths):
            pass
        def create(self, path):
            return Zero()
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (
        SamplingConditionGroup('ads', 'gcmc_md', ('A',), temperatures_k=(303.,),
                               pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                                   (1,), moves=2, every=100)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
                                      (SamplingSeedSource('A', (host,)),),
                                      OrderedSamplingModels((model,), ('member-0',)), None)
    backend = GcmcSamplingNativeOps(calculator_provider=Provider(), guest_sources={'ads': guest_path},
                                   gcmc_class=official, device='cpu')
    backend.execute(SamplingExecutionRequest(evidence))
    assert backend.validate_current(evidence).complete
    assert backend.open_current(evidence).source.catalog() == ()
    (tmp_path/'sample/tasks/task_000000/final_structure.extxyz').write_text('damaged')
    assert not backend.validate_current(evidence).complete


def test_selected_lammps_factory_routes_mc_worker_with_same_ordered_models(tmp_path, official, monkeypatch):
    from types import SimpleNamespace
    from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory
    from workflow_backends.sampling import SamplingBackendBinding, SamplingEvidenceRequest, SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.common.gcmc import GcmcSamplingParameters
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.structures.formats import write_extxyz_structures
    from workflow_runtime.lifecycle.environment import OperationContext
    from workflow_backends.gcmc.worker import execute_manifest
    import workflow_backends.gcmc.runtime as runtime

    host, guest, pressure = inputs()
    guest_path, model = tmp_path/'guest.extxyz', tmp_path/'model.fixture'
    write_extxyz_structures(guest_path, [guest])
    model.write_text('fixture')
    context = OperationContext('sampler.lammps.pt', 'test', (), {'python': '/test/python'}, {})
    class Provider:
        def preflight(self, paths):
            assert paths == (model,)
        def create(self, path):
            assert path == model
            return Zero()
    calls = []
    def operation(selected, executable, args, **kwargs):
        assert selected == context and executable == 'python'
        assert args[:3] == ('-m', 'workflow_backends.gcmc.worker', '--manifest')
        calls.append(args[-1])
        execute_manifest(args[-1], calculator_provider=Provider(), gcmc_class=official)
    monkeypatch.setattr(runtime, 'run_operation', operation)
    backend = LammpsSamplingBackendFactory(SimpleNamespace(context=context)).bind(
        SamplingBackendBinding((), ('He', 'Ne'), {}, guest_inputs={'ads': guest_path}))
    plan = ExplicitSamplingPlan((SeedRole('A', 'joint'),), (
        SamplingConditionGroup('ads', 'gcmc_md', ('A',), temperatures_k=(303.,),
                               pressures_bar=(pressure,), gcmc=GcmcSamplingParameters(
                                   (2,), moves=137, every=20)),), 1)
    evidence = SamplingEvidenceRequest(tmp_path/'run.json', 0, 0, tmp_path/'sample', plan,
                                      (SamplingSeedSource('A', (host,)),),
                                      OrderedSamplingModels((model,), ('member-0',)), None)
    assert backend.execute(SamplingExecutionRequest(evidence)).task_count == 1
    assert len(calls) == 1
    assert backend.validate_current(evidence).complete
    assert len(backend.open_current(evidence).source.catalog()) == 13


def test_installed_official_kernel_is_verified_before_loading(official):
    from workflow_backends.gcmc.upstream import load_official
    assert load_official() is official


def test_phase_state_preserves_ase_string_labels_without_pickle(tmp_path):
    from ase import Atoms
    from workflow_backends.gcmc.state import save_state, load_state
    atoms = Atoms('HH', cell=[8, 8, 8], pbc=True)
    atoms.set_array('site_role', np.array(['framework', 0], dtype=object))
    atoms.set_velocities([[.01, .02, .03], [.04, .05, .06]])
    path = tmp_path/'state.json'
    save_state(path, {'atoms': atoms})
    recovered = load_state(path)['atoms']
    for key in atoms.arrays:
        np.testing.assert_array_equal(atoms.arrays[key], recovered.arrays[key])
    atoms.arrays['site_role'][0] = object()
    with pytest.raises(ValueError, match='unsupported plain phase state'):
        save_state(tmp_path/'invalid.json', {'atoms': atoms})


def test_public_sample_preflight_requires_guest_but_not_a_pure_mc_md_recipe(tmp_path):
    from ase import Atoms
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch, prepare_execution_block
    from workflow_core.structures.formats import write_extxyz_structures
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz'}}, 'round_count': 1,
        'condition_groups': {'ads': {'kind': 'gcmc_md', 'temperatures_k': [303],
            'cycles': 1, 'mc': {'guest': 'data/guest.xyz', 'pressures_bar': [.1],
                              'moves': 10, 'eos': {'kind': 'ideal'}}}},
    })
    write_extxyz_structures(tmp_path/'data/A.xyz', [Atoms('H', cell=[8, 8, 8], pbc=True)])
    launch = prepare_lifecycle_launch(run)
    with pytest.raises(ValueError, match='mc.guest'):
        prepare_execution_block(launch, block_id='r0/sample', require_environment=False)
    write_extxyz_structures(tmp_path/'data/guest.xyz', [Atoms('H')])
    prepare_execution_block(launch, block_id='r0/sample', require_environment=False)
    invalid = Atoms('H', cell=[8, 8, 8], pbc=True)
    invalid.set_array('dice_atom_id', np.array([1]))
    write_extxyz_structures(tmp_path/'data/A.xyz', [invalid])
    with pytest.raises(ValueError, match='identity|membership'):
        prepare_execution_block(launch, block_id='r0/sample', require_environment=False)
    invalid = Atoms('H', cell=[8, 8, 8], pbc=True)
    invalid.set_masses([0.])
    write_extxyz_structures(tmp_path/'data/A.xyz', [invalid])
    with pytest.raises(ValueError, match='mass'):
        prepare_execution_block(launch, block_id='r0/sample', require_environment=False)
    write_extxyz_structures(tmp_path/'data/A.xyz', [Atoms('H')])
    with pytest.raises(ValueError, match='periodic'):
        prepare_execution_block(launch, block_id='r0/sample', require_environment=False)
