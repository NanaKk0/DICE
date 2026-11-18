"""Result transport for ordinary native protocols, independent of MD budgets."""
from dataclasses import asdict
from io import StringIO
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read

from workflow_backends.common.frame_index import FrameByteIndex, file_snapshot
from workflow_backends.common.recipe_inputs import contained_artifact
from workflow_backends.observation_source import ObservationLocator, ObservationRecord
from workflow_backends.lammps.observation_source import _file_digest, _max_deviation_rows


def scan_protocol_dump(path, *, type_map, seed, variable_types=False, variable_count=False, retain_prefix=False, allow_empty=False):
    """One streaming validation pass also locates every complete frame."""
    path = Path(path)
    snapshot = file_snapshot(path)
    spans, rows = [], []
    expected_ids = None if variable_count else np.column_stack((
        np.arange(1, len(seed) + 1),
        [type_map.index(symbol) + 1 for symbol in seed.get_chemical_symbols()]))
    with path.open('rb') as stream:
        while True:
            try:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip() != b'ITEM: TIMESTEP':
                    raise ValueError('native protocol dump has an invalid frame boundary')
                step = int(stream.readline())
                if stream.readline().strip() != b'ITEM: NUMBER OF ATOMS':
                    raise ValueError('native protocol dump is missing atom count')
                count = int(stream.readline())
                if count <= 0 or (not variable_count and count != len(seed)):
                    raise ValueError('native protocol atom count changed')
                header = stream.readline().split()
                if header[:3] != [b'ITEM:', b'BOX', b'BOUNDS']:
                    raise ValueError('native protocol dump is missing cell')
                box = []
                for _ in range(3):
                    bounds = np.fromstring(stream.readline().decode(), sep=' ')
                    if len(bounds) not in {2, 3} or not np.isfinite(bounds).all():
                        raise ValueError('native protocol cell is invalid')
                    box.append(bounds)
                flags = header[-3:]
                if len(header) not in {6, 9} or any(flag not in {
                    b'pp', b'ff', b'fs', b'sf', b'fm', b'mf', b'mm', b'ms', b'sm', b'ss'} for flag in flags):
                    raise ValueError('native protocol cell boundary flags are invalid')
                lengths = [row[1] - row[0] for row in box]
                if len(header) == 9:
                    if header[3:6] != [b'xy', b'xz', b'yz'] or any(len(row) != 3 for row in box):
                        raise ValueError('native protocol cell tilt is invalid')
                    xy, xz, yz = (row[2] for row in box)
                    lengths[0] -= max(0, xy, xz, xy + xz) - min(0, xy, xz, xy + xz)
                    lengths[1] -= abs(yz)
                elif any(len(row) != 2 for row in box):
                    raise ValueError('native protocol cell bounds are invalid')
                if any(value <= 0 for value in lengths):
                    raise ValueError('native protocol cell must be right-handed and full-rank')
                columns = stream.readline().decode().split()
                if columns[:2] != ['ITEM:', 'ATOMS']:
                    raise ValueError('native protocol atom header is invalid')
                columns = columns[2:]
                required = ['id', 'type', 'x', 'y', 'z']
                if len(set(columns)) != len(columns) or any(name not in columns for name in required):
                    raise ValueError('native protocol dump is missing identity/geometry columns')
                indices = [columns.index(name) for name in required]
                atom_rows = [stream.readline() for _ in range(count)]
                if any(len(row.split()) != len(columns) for row in atom_rows):
                    raise ValueError('native protocol dump has a truncated atom row')
                values = np.fromstring(b''.join(atom_rows).decode('ascii'), sep=' ')
                if values.size != count * len(columns):
                    raise ValueError('native protocol dump has a nonnumeric atom row')
                data = values.reshape(count, len(columns))[:, indices]
                ids, types = data[:, 0], data[:, 1]
                if (not np.isfinite(data).all() or not np.equal(ids, ids.astype(np.int64)).all()
                    or len(set(ids)) != count or np.any(ids <= 0) or np.any(types < 1)
                    or np.any(types > len(type_map)) or not np.equal(types, types.astype(np.int64)).all()):
                    raise ValueError('native protocol atom identity/geometry is invalid')
                identity = np.column_stack((ids, types))
                if expected_ids is None:
                    expected_ids = identity
                elif not variable_count and not np.array_equal(identity[:, :1 if variable_types else 2], expected_ids[:, :1 if variable_types else 2]):
                    raise ValueError('native protocol changed atom identity')
                spans.append((start, stream.tell()))
                rows.append({'ordinal': len(rows), 'native_step': step, 'atom_count': count})
            except (ValueError, UnicodeError, IndexError, OverflowError):
                if not retain_prefix:
                    raise
                break
    if file_snapshot(path) != snapshot:
        raise ValueError('native protocol changed during result validation')
    if not rows and not allow_empty:
        raise ValueError('native protocol produced no observation geometry')
    return tuple(rows), FrameByteIndex(path, snapshot, tuple(spans))


def validate_protocol_result(root, *, intent=None, seed=None, type_map=None, member_ids=None,
                             digest_cache=None):
    root = Path(root)
    cache = {} if digest_cache is None else digest_cache
    context = json.loads((root/'task_context.json').read_text())
    result = json.loads((root/'protocol_result.json').read_text())
    if (context.get('schema') != 'dice.native-protocol-task' or context.get('revision') != 1
        or result.get('schema') != 'dice.native-protocol-result' or result.get('revision') not in {1,2,3}
        or result.get('status') not in ({'completed','failed'} if result.get('revision') == 3 else {'completed'})
        or result.get('task_id') != root.name or context.get('task_id') != root.name
        or result.get('context_digest') != _file_digest(root/'task_context.json', cache=cache)):
        raise ValueError('native protocol result/context is incompatible')
    if intent is not None:
        for name in ('task_id', 'sampling_role_id', 'selection_group_id', 'condition_group_id', 'source_frame_index'):
            if context[name] != getattr(intent, name):
                raise ValueError(f'native protocol {name} changed')
        if context['request'] != asdict(intent.protocol_request):
            raise ValueError('native protocol scientific request changed')
    if member_ids is not None and context['member_ids'] != list(member_ids):
        raise ValueError('native protocol model membership changed')
    if type_map is not None and set(context['type_map']) != set(type_map):
        raise ValueError('native protocol element map changed')
    budget=context['effective']['total_steps']
    failed = result['status'] == 'failed'
    if failed:
        failure = json.loads((root/'native_protocol_failure.json').read_text())
        if (context['observation_kind'] != 'md' or failure.get('status') != 'failed'
            or failure.get('task_id') != root.name or not isinstance(failure.get('error'), str)):
            raise ValueError('native protocol failed-prefix evidence is incompatible')
    else:
        terminal = (root/'native_protocol_complete.out').read_text().split()
        if (len(terminal) != 2 or terminal[0] != 'DICE_NATIVE_PROTOCOL_COMPLETE'
            or not terminal[1].isdigit() or int(terminal[1]) != result.get('terminal_native_step')):
            raise ValueError('native protocol terminal evidence is incomplete')
    if not failed and context['observation_kind']=='md' and budget is not None and result.get('terminal_native_step')!=budget:
        raise ValueError('native protocol step budget is incomplete')
    required={'seed.xyz','conf.lmp','protocol.lammps','input.lammps','dice_task_setup.lammps',context['trajectory']}
    required.update(context.get('inputs', ()))
    models={name for name in result['files'] if name.startswith('graph_')}
    if not required<=set(result['files']) or len(models)!=len(context['member_ids']):
        raise ValueError('native protocol canonical file inventory is incomplete')
    if result['revision'] >= 2:
        required.add('native_protocol_failure.json' if failed else 'native_protocol_complete.out')
        if context['observation_kind'] == 'optimization':
            required.add('protocol_endpoint.dump')
        if len(context['member_ids']) > 1:
            required.add(context.get('deviations', 'model_devi.out'))
        suffixes={Path(name).suffix for name in models}
        if len(suffixes) != 1 or not suffixes <= {'.pb','.pt2'}:
            raise ValueError('native protocol model ABI inventory is incompatible')
        suffix=next(iter(suffixes))
        if (models != {f'graph_{i:03d}{suffix}' for i in range(len(context['member_ids']))}
            or set(result['files']) != required | models):
            raise ValueError('native protocol canonical file inventory is not exact')
    for name, digest in result['files'].items():
        if _file_digest(contained_artifact(root, name), cache=cache) != digest:
            raise ValueError(f'native protocol artifact changed: {name}')
    from workflow_core.structures.formats import read_extxyz_structures
    from workflow_core.structures.exact_identity import exact_structure_hash
    stored, = read_extxyz_structures(root/'seed.xyz', allow_empty=False)
    if seed is not None and exact_structure_hash(seed) != exact_structure_hash(stored):
        raise ValueError('native protocol seed changed')
    rows, index = scan_protocol_dump(root/context['trajectory'], type_map=context['type_map'], seed=stored,
        variable_types=context['variable_types'], variable_count=context['variable_count'], allow_empty=failed)
    if list(rows) != result['frames']:
        raise ValueError('native protocol observation inventory changed')
    return context, result, index


class NativeProtocolObservationSource:
    def __init__(self, *, run_json_path, iter_index, round_index, tasks):
        namespace = f'{Path(run_json_path).resolve()}\0{iter_index}\0{round_index}\0'
        self._frames, records = {}, []
        self.failed_task_count = 0
        cache = {}
        for root, intent in tasks:
            context, result, index = validate_protocol_result(root, intent=intent, digest_cache=cache)
            self.failed_task_count += result['status'] == 'failed'
            deviations = _max_deviation_rows(Path(root)/context.get('deviations', 'model_devi.out')) if len(context['member_ids']) > 1 and result['frames'] else {}
            for row in result['frames']:
                # Native MD step zero remains a seed diagnostic. Optimizers can
                # validly converge at step zero and expose their endpoint.
                if row['native_step'] == 0 and context['observation_kind'] == 'md':
                    continue
                ordinal, step = row['ordinal'], row['native_step']
                identity = 'sha256:' + hashlib.sha256((namespace + intent.task_id + '\0' + str(ordinal)).encode()).hexdigest()
                uncertainty = deviations.get(step)
                if len(context['member_ids']) > 1 and uncertainty is None:
                    raise ValueError('native protocol observation lacks committee evidence')
                records.append(ObservationRecord(identity, intent.selection_group_id,
                    int(intent.task_id.split('_')[-1]), ordinal,
                    step if context['observation_kind'] == 'md' else None, uncertainty,
                    ObservationLocator('LAMMPS-PROTOCOL', int(intent.task_id.split('_')[-1]), ordinal, step),
                    sampling_role_id=intent.sampling_role_id, selection_group_id=intent.selection_group_id,
                    condition_group_id=intent.condition_group_id, task_id=intent.task_id,
                    optimizer_step=step if context['observation_kind'] == 'optimization' else None))
                self._frames[identity] = (index, ordinal, tuple(context['type_map']))
        self._records = tuple(records)
        self._records_by_id = {record.structure_id: record for record in records}
        self.allows_empty = not records and self.failed_task_count > 0
        if not records and not self.allows_empty:
            raise ValueError('native protocol has no screen-visible observations')

    def catalog(self):
        return self._records

    def replay(self, identities, *, chunk_size):
        identities = tuple(identities)
        if type(chunk_size) is not int or chunk_size <= 0 or len(set(identities)) != len(identities):
            raise ValueError('native protocol replay requires unique identities and positive chunk size')
        by_id = self._records_by_id
        for start in range(0, len(identities), chunk_size):
            wanted, loaded, groups = identities[start:start+chunk_size], {}, {}
            for identity in wanted:
                if identity not in self._frames:
                    raise ValueError('native protocol replay identity is absent')
                index, ordinal, types = self._frames[identity]
                groups.setdefault(index.path, (index, types, []))[2].append((identity, ordinal))
            for index, types, items in groups.values():
                for (identity, ordinal), raw in zip(items, index.read(i for _, i in items), strict=True):
                    atoms = read(StringIO(raw.decode()), format='lammps-dump-text', specorder=list(types))
                    row = by_id[identity]
                    atoms.info.update(structure_id=identity, system_id=row.system_id, structure_source='workflow',
                        sampling_role_id=row.sampling_role_id, selection_group_id=row.selection_group_id,
                        condition_group_id=row.condition_group_id, task_id=row.task_id, md_step=row.md_step)
                    if row.optimizer_step is not None:
                        atoms.info['optimizer_step'] = row.optimizer_step
                    if row.student_uncertainty is not None:
                        atoms.info['student_uncertainty'] = row.student_uncertainty
                    loaded[identity] = atoms
            yield tuple(loaded[identity] for identity in wanted)
