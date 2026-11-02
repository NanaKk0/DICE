"""Plain native request entry; no model choice or lifecycle sequencing here."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps
from workflow_backends.gcmc.state import load_state
from workflow_backends.gcmc.upstream import load_official
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.gcmc import GcmcMdParameters, GcmcSamplingParameters
from workflow_core.common.sampling_plan import ResolvedSamplingTask
from workflow_core.structures.formats import read_extxyz_structures


class DeepMDCalculatorProvider:
    # Existing Step 2 DeepMD/LAMMPS numerical energy parity bound. This is
    # provider-internal validation, not an MC energy correction or run field.
    reference_tolerance_ev = 2e-3
    def preflight(self, paths):
        from deepmd.calculator import DP
        del DP
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError('GCMC requires the prepared ordered DeepMD models')

    def create(self, path):
        from deepmd.calculator import DP
        return DP(model=path)


def execute_manifest(path, *, calculator_provider=None, gcmc_class=None):
    payload = load_state(path)
    expected = {'schema', 'revision', 'run_json_path', 'iter', 'round', 'sample_dir',
                'models', 'member_ids', 'type_map', 'device', 'operation_role',
                'operation_profile_id', 'tasks'}
    if set(payload) not in (expected, expected | {'md_transport'}) or payload['schema'] != 'dice.native.gcmc-request' or payload['revision'] != 1:
        raise ValueError('unsupported GCMC native request')
    if payload['operation_role'] not in {'sampler.lammps.tf', 'sampler.lammps.pt'}:
        raise ValueError('NEP GCMC backend is deferred; only DeepMD is supported')
    if payload['device'] not in {'cpu', 'cuda'}:
        raise ValueError('unsupported GCMC device')
    provider = calculator_provider or DeepMDCalculatorProvider()
    upstream = gcmc_class or load_official()
    models = OrderedSamplingModels(tuple(Path(p) for p in payload['models']), tuple(payload['member_ids']))
    provider.preflight(models.paths)
    evidence = SimpleNamespace(run_json_path=Path(payload['run_json_path']), iter_index=payload['iter'],
                               round_index=payload['round'], sample_dir=Path(payload['sample_dir']), models=models)
    prepared = []
    for item in payload['tasks']:
        if set(item) != {'task', 'seed', 'guest_path'}:
            raise ValueError('malformed GCMC worker task')
        raw = dict(item['task'])
        parameters = dict(raw.pop('gcmc'))
        if parameters.get('md') is not None:
            parameters['md'] = GcmcMdParameters(**parameters['md'])
        for key in ('cycles_by_round', 'weights'):
            parameters[key] = tuple(parameters[key])
        parameters['radii_a'] = tuple(tuple(pair) for pair in parameters['radii_a'])
        task = ResolvedSamplingTask(**raw, gcmc=GcmcSamplingParameters(**parameters))
        if task.kind != 'gcmc_md' or task.program is not None or task.neb is not None or task.relax is not None:
            raise ValueError('GCMC worker only accepts closed GCMC tasks')
        guests = read_extxyz_structures(item['guest_path'])
        if len(guests) != 1:
            raise ValueError('GCMC worker requires one guest template')
        if not set(item['seed'].get_chemical_symbols()+guests[0].get_chemical_symbols()) <= set(payload['type_map']):
            raise ValueError('GCMC species are outside the selected type map')
        prepared.append((task, item['seed'], guests[0]))
    if not prepared or len({t.task_id for t, _, _ in prepared}) != len(prepared):
        raise ValueError('GCMC worker requires unique nonempty tasks')
    (evidence.sample_dir/'tasks').mkdir(parents=True, exist_ok=True)
    md_runners = {}
    if any(task.gcmc.md is not None for task, _, _ in prepared):
        from workflow_backends.gcmc.lammps import LammpsSegmentRunner
        transport = payload.get('md_transport')
        if not isinstance(transport, dict) or set(transport) != {'executable', 'mode', 'recipes'}:
            raise ValueError('hybrid worker requires resolved native MD transport')
        for task, _, _ in prepared:
            if task.gcmc.md is not None:
                md_runners[task.condition_group_id] = LammpsSegmentRunner(
                    recipe=transport['recipes'][task.condition_group_id], models=models.paths,
                    type_map=payload['type_map'], mode=transport['mode'], executable=transport['executable'])
    backend = GcmcSamplingNativeOps(calculator_provider=provider, guest_sources={},
                                   gcmc_class=upstream, device=payload['device'],
                                   md_runners=md_runners,
                                   reference_tolerance_ev=getattr(provider, 'reference_tolerance_ev', 0.))
    for task, seed, guest in prepared:
        backend._execute_task(evidence, task, seed, guest, models.paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    execute_manifest(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
