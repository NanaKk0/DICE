"""Selected-environment Widom task entry."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from workflow_backends.gcmc.state import load_state
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.sampling_plan import ResolvedSamplingTask
from workflow_core.common.widom import WidomSamplingParameters
from workflow_core.structures.formats import read_extxyz_structures
from workflow_core.structures.widom import validate_inputs
from workflow_backends.widom.adapter import WidomSamplingNativeOps
from workflow_backends.widom.evaluation import DeepMDEvaluationProvider


def execute_manifest(path, *, provider=None):
    payload = load_state(path)
    if payload['schema'] != 'dice.native.widom-request' or payload['revision'] != 1:
        raise ValueError('unsupported widom native request')
    if payload['operation_role'] not in {'sampler.lammps.tf', 'sampler.lammps.pt'}:
        raise ValueError('widom currently supports DeepMD only')
    models = OrderedSamplingModels(tuple(Path(p) for p in payload['models']), tuple(payload['member_ids']))
    raw = dict(payload['task']); raw['widom'] = WidomSamplingParameters(**raw['widom'])
    task = ResolvedSamplingTask(**raw)
    if task.kind != 'widom' or task.gcmc is not None or task.neb is not None or task.relax is not None:
        raise ValueError('widom worker requires closed widom task')
    guests = read_extxyz_structures(payload['guest_path'])
    validate_inputs(payload['seed'], guests, task.widom, payload['type_map'])
    provider = provider or DeepMDEvaluationProvider()
    provider.preflight(models.paths, payload['type_map'])
    evidence = SimpleNamespace(iter_index=payload['iter'], round_index=payload['round'],
        sample_dir=Path(payload['sample_dir']), model_paths=models.paths, models=models)
    backend = WidomSamplingNativeOps(provider=provider, guest_sources={}, type_map=payload['type_map'])
    backend.execute_task(evidence, task, payload['seed'], guests, staging_root=payload['staging_root'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True)
    execute_manifest(parser.parse_args().manifest)


if __name__ == '__main__': main()
