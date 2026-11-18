"""Explicit native file bindings; discovery never scans a user's directory."""
from pathlib import Path, PurePosixPath
import shutil


_OWNED = {'seed.xyz', 'conf.lmp', 'model.xyz', 'input.lammps', 'run.in',
          'protocol.lammps', 'protocol.gpumd', 'dice_task_setup.lammps',
          'task_context.json', 'protocol_result.json', 'gpumd_protocol_result.json',
          'native_protocol_complete.out', 'protocol_endpoint.dump', 'traj.lammpstrj',
          'model_devi.out', 'active.xyz', 'active.out', 'native_protocol_failure.json',
          'committed_prefix.lammpstrj', 'committed_model_devi.out'}


def contained_artifact(root, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name
        or any(part in {'', '.', '..'} for part in name.split('/'))):
        raise ValueError('native artifact requires a contained relative name')
    destination = Path(root)/name
    if any(part.is_symlink() for part in (destination, *destination.parents)
           if part != Path(root) and Path(root) in part.parents):
        raise ValueError('native artifact path contains a symlink')
    return destination


def validate_recipe_inputs(inputs):
    for name, source in inputs.items():
        contained_artifact(Path('/native-task'), name)
        top = name.split('/')[0]
        if top in _OWNED or top.startswith(('graph_', 'nep_', 'dice_')):
            raise ValueError(f'native input conflicts with task-owned output: {name}')
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'native input must be an existing regular file: {source}')


def stage_recipe_inputs(root, inputs):
    validate_recipe_inputs(inputs)
    for name, source in inputs.items():
        destination = contained_artifact(root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tuple(inputs)
