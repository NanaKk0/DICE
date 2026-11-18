"""Execute clean native protocols through the existing sampling execution pool."""
from dataclasses import asdict, dataclass, field
import inspect
import json
import math
from pathlib import Path
import shutil

from ase.data import atomic_masses, atomic_numbers

from workflow_backends.common.model_files import link_model_file
from workflow_backends.common.seed_selection import publish_seed_selection_manifest
from workflow_backends.common.recipe_inputs import stage_recipe_inputs
from workflow_backends.common.task_outcomes import TaskOutcome
from workflow_backends.lammps.task_pool import execute_lammps_tasks
from workflow_backends.lammps.execution_mode import select_lammps_execution_mode
from workflow_backends.lammps.native_protocol import compile_lammps_protocol
from workflow_backends.lammps.observation_source import _file_digest
from workflow_backends.lammps.protocol import command_spans, _PREAMBLE
from workflow_backends.lammps.protocol_result import (
    NativeProtocolObservationSource, scan_protocol_dump, validate_protocol_result,
)
from workflow_backends.sample_artifacts import canonical_selected_seed, publish_selected_seeds
from workflow_backends.sampling import SamplingExecutionResult, SamplingObservationResult, SamplingValidationResult
from workflow_core.structures.formats import atoms_to_lammps_data, write_extxyz_structures


def load_protocol_source(path, *, visited=()):
    """Expand literal includes at their source location, with explicit cycle checks."""
    path = Path(path).resolve()
    if path in visited or len(visited) > 32:
        raise ValueError('native protocol include cycle or excessive nesting')
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    for start, end, row in reversed(command_spans(lines)):
        if row[0] == 'include' and row != ['include', 'dice_task_setup.lammps']:
            if len(row) != 2 or '$' in row[1]:
                raise ValueError('native include needs a resolvable literal path')
            dependency = path.parent/row[1]
            lines[start:end] = [load_protocol_source(dependency, visited=(*visited,path)) + '\n']
    return ''.join(lines)


def bind_protocol_setup(text):
    """Bind seed/model once while retaining native masses and physical commands."""
    lines = text.splitlines(keepends=True)
    commands = command_spans(lines)
    structures = [item for item in commands if item[2][0] in {'read_data','read_restart','create_box'}]
    if len(structures) > 1 or any(row[0] in {'clear','read_restart','create_box'} for _,_,row in commands):
        raise ValueError('native protocol requires one seed binding without state recreation')
    explicit = [item for item in commands if item[2] == ['include','dice_task_setup.lammps']]
    if len(explicit) > 1 or (explicit and structures):
        raise ValueError('native protocol seed binding is ambiguous')
    edits = {}
    for start,end,row in commands:
        if row[0] == 'read_data':
            edits[start]=(end,'include dice_task_setup.lammps\n')
        elif row[0] == 'pair_style':
            if len(row) < 2 or row[1] not in {'deepmd','deepmd/kk'}:
                raise ValueError('native protocol model binding requires a DeepMD pair style')
            edits[start]=(end,'')
        elif row[0] == 'pair_coeff':
            edits[start]=(end,'')
    if not structures and not explicit:
        insertion = next((start for start,_,row in commands if row[0] not in _PREAMBLE),len(lines))
        edits[insertion]=(insertion,'include dice_task_setup.lammps\n')
    for start,(end,value) in sorted(edits.items(),reverse=True):
        lines[start:end]=[value]
    return ''.join(lines)


def resolve_native_type_map(text, active_type_map):
    """Keep native type-number meaning when rebinding a complete DeepMD input."""
    coefficients = [row for _, _, row in command_spans(text.splitlines(keepends=True))
                    if row[0] == 'pair_coeff']
    if not coefficients:
        return tuple(active_type_map)
    if len(coefficients) != 1 or coefficients[0][1:3] != ['*', '*']:
        raise ValueError('native model binding requires one pair_coeff * * element map')
    symbols = coefficients[0][3:]
    if not symbols:  # DeePMD's implicit map follows the workflow active order.
        return tuple(active_type_map)
    if len(set(symbols)) != len(symbols) or set(symbols) != set(active_type_map):
        raise ValueError('native pair_coeff element map must cover the active elements once')
    return tuple(symbols)


@dataclass
class NativeLammpsProtocolOps:
    executor: object
    type_map: tuple[str,...]
    group_recipes: dict[str,Path]
    group_inputs: dict = field(default_factory=dict)

    @staticmethod
    def _tasks(evidence):
        return tuple(task for task in evidence.plan.tasks_for_round(evidence.round_index)
                     if task.protocol_request is not None)

    def _source(self,evidence):
        return NativeProtocolObservationSource(run_json_path=evidence.run_json_path,
            iter_index=evidence.iter_index,round_index=evidence.round_index,
            tasks=tuple((evidence.sample_dir/'tasks'/task.task_id,task) for task in self._tasks(evidence)))

    def open_current(self,evidence):
        return SamplingObservationResult(self._source(evidence))

    def validate_current(self,evidence):
        try:
            cache={}
            for task in self._tasks(evidence):
                context,result,_index=validate_protocol_result(evidence.sample_dir/'tasks'/task.task_id,intent=task,
                    seed=evidence.seed_mapping[task.sampling_role_id][0],type_map=self.type_map,
                    member_ids=evidence.models.member_ids,digest_cache=cache)
            return SamplingValidationResult(bool(self._tasks(evidence)))
        except (OSError,ValueError,KeyError,TypeError):
            return SamplingValidationResult(False)

    def execute(self,request):
        evidence=request.evidence
        tasks=self._tasks(evidence)
        context=getattr(self.executor,'context',None)
        cache={}
        seeds={role:canonical_selected_seed(frames[0],system_id=role)
               for role,frames in evidence.seed_mapping.items()}
        publish_selected_seeds(evidence.sample_dir/'seeds.xyz',tuple(seeds.values()))
        if evidence.seed_selections:
            publish_seed_selection_manifest(evidence.sample_dir/'seed_selection.json',
                iter_index=evidence.iter_index, round_index=evidence.round_index,
                selections=evidence.seed_selections)
        prepared=[]
        for task in tasks:
            source=load_protocol_source(self.group_recipes[task.condition_group_id])
            native_type_map = resolve_native_type_map(source, self.type_map)
            resolved=compile_lammps_protocol(source,task.protocol_request)
            bound=bind_protocol_setup(resolved.text)
            root=evidence.sample_dir/'tasks'/task.task_id
            root.mkdir(parents=True,exist_ok=True)
            inputs = stage_recipe_inputs(root, self.group_inputs.get(task.condition_group_id, {}))
            seed=seeds[task.sampling_role_id]
            write_extxyz_structures(root/'seed.xyz',(seed,),allow_empty=False)
            atoms_to_lammps_data(seed,root/'conf.lmp',type_map=list(native_type_map))
            kinds={getattr(model,'deepmd_recipe_kind','standard') for model in getattr(evidence.models,'models',())}
            recipe_kind=next(iter(kinds)) if len(kinds)==1 else 'standard'
            cli='pt' if evidence.model_paths[0].suffix=='.pt2' else 'tf'
            mode=select_lammps_execution_mode(sampler_role=f'sampler.lammps.{cli}',
                deepmd_recipe_kind=recipe_kind,model_count=len(evidence.model_paths))
            names=[]
            for index,model in enumerate(evidence.model_paths):
                name=f'graph_{index:03d}{model.suffix}'
                link_model_file(model,root/name);names.append(name)
            rows=command_spans(bound.splitlines(keepends=True))
            ids={row[1] for _,_,row in rows if row[0]=='dump' and len(row)>1}
            dump_id='dice_observations'
            while dump_id in ids:dump_id+='_'
            trajectory='traj.lammpstrj'
            while any(trajectory in row for _,_,row in rows):trajectory='dice_'+trajectory
            pair=f'pair_style {"deepmd/kk" if mode=="kokkos" else "deepmd"} '+ ' '.join(names)
            if len(names)>1:pair+=f' out_freq {resolved.cadence} out_file model_devi.out'
            # PT2 graph lowers resolve ghost owners in ordinary LAMMPS too.
            setup=(['atom_modify map yes'] if mode=='kokkos' or cli=='pt' else [])+['read_data conf.lmp']
            setup += [f'mass {i} {atomic_masses[atomic_numbers[symbol]]:.17g}'
                      for i,symbol in enumerate(native_type_map,1)]
            setup += [pair,'pair_coeff * * '+' '.join(native_type_map)]
            observations = (f'dump {dump_id} all custom {resolved.cadence} {trajectory} id type x y z\n'
                            f'dump_modify {dump_id} sort id format float %.17g\n')
            # LAMMPS forbids ortho/triclinic conversion with an active dump.
            # Bind geometry/model first, then attach observations at execution.
            execution = next((start for start, _, row in rows if row[0] in {'run', 'minimize'}), None)
            if execution is None:
                raise ValueError('native protocol needs an observable run or minimize command')
            insertion = next((start for start, _, row in rows if row[0] == 'label' and start < execution), execution)
            lines = bound.splitlines(keepends=True)
            lines[insertion:insertion] = [observations]
            bound = ''.join(lines)
            (root/'dice_task_setup.lammps').write_text('\n'.join(setup)+'\n')
            (root/'protocol.lammps').write_text(source)
            observation_kind='optimization' if any(row[0]=='minimize' for _,_,row in rows) else 'md'
            endpoint = ('\nwrite_dump all custom protocol_endpoint.dump id type x y z '
                        'modify sort id format float %.17g\n') if observation_kind == 'optimization' else ''
            (root/'input.lammps').write_text(bound+endpoint+'\nprint "DICE_NATIVE_PROTOCOL_COMPLETE $(step:%.0f)" file native_protocol_complete.out\n')
            variable_count=any(row[0] in {'delete_atoms','create_atoms'} or (row[0]=='fix' and len(row)>3 and row[3]=='gcmc') for _,_,row in rows)
            variable_types=any(row[0]=='fix' and len(row)>3 and row[3]=='atom/swap' for _,_,row in rows)
            payload=dict(schema='dice.native-protocol-task',revision=1,task_id=task.task_id,
                sampling_role_id=task.sampling_role_id,selection_group_id=task.selection_group_id,
                condition_group_id=task.condition_group_id,source_frame_index=task.source_frame_index,
                request=asdict(task.protocol_request),member_ids=list(evidence.models.member_ids),
                type_map=list(native_type_map),trajectory=trajectory,observation_kind=observation_kind,
                variable_types=variable_types,variable_count=variable_count,inputs=list(inputs),
                effective=dict(ensemble=resolved.ensemble,temperature_k=resolved.temperature_k,
                    pressure_bar=resolved.pressure_bar,total_steps=resolved.total_steps,timestep_ps=resolved.timestep_ps,
                    cadence=resolved.cadence,settings=resolved.settings))
            (root/'task_context.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
            native=dict(type_map=native_type_map,model_count=len(names),deepmd_recipe_kind=recipe_kind,
                sampler_role=f'sampler.lammps.{cli}',execution_mode=mode,kind='native',trajectory_path=trajectory,
                steps=resolved.total_steps,cadence=resolved.cadence)
            prepared.append((task,root,native,payload,tuple(names),seed))

        def retain_failure(prepared, error):
            task,root,native,payload,names,seed=prepared
            if payload['observation_kind'] != 'md':
                raise error
            path = root/payload['trajectory']
            if path.exists():
                rows,index=scan_protocol_dump(path,type_map=payload['type_map'],seed=seed,
                    variable_types=payload['variable_types'],variable_count=payload['variable_count'],
                    retain_prefix=True,allow_empty=True)
            else:
                rows,index=(),None
            deviations = {}
            if len(names)>1 and (root/'model_devi.out').is_file():
                with (root/'model_devi.out').open() as stream:
                    for line in stream:
                        if not line.strip() or line.lstrip().startswith('#'): continue
                        try:
                            values=[float(value) for value in line.split()]
                            if len(values)<7 or not all(math.isfinite(value) for value in values) or values[0]!=int(values[0]) or values[4]<0:
                                break
                            step=int(values[0])
                            if step in deviations: break
                            deviations[step]=line
                        except (ValueError,OverflowError): break
                count=0
                for row in rows:
                    if row['native_step'] > 0 and row['native_step'] not in deviations: break
                    count+=1
                rows=rows[:count]
            prefix='committed_prefix.lammpstrj'
            with (root/prefix).open('wb') as output:
                if rows:
                    for raw in index.read(range(len(rows))):
                        output.write(raw)
            payload={**payload,'trajectory':prefix}
            files=['seed.xyz','conf.lmp','protocol.lammps','input.lammps','dice_task_setup.lammps',
                   prefix,'native_protocol_failure.json',*names,*payload['inputs']]
            if len(names)>1:
                payload['deviations']='committed_model_devi.out'
                (root/payload['deviations']).write_text(''.join(deviations[row['native_step']] for row in rows if row['native_step'] in deviations))
                files.append(payload['deviations'])
            (root/'task_context.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
            (root/'native_protocol_failure.json').write_text(json.dumps(dict(status='failed',task_id=task.task_id,error=str(error)))+'\n')
            result=dict(schema='dice.native-protocol-result',revision=3,status='failed',task_id=task.task_id,
                context_digest=_file_digest(root/'task_context.json',cache=cache),frames=rows,
                terminal_native_step=rows[-1]['native_step'] if rows else None,
                files={name:_file_digest(root/name,cache=cache) for name in files})
            (root/'protocol_result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
            validate_protocol_result(root,intent=task,digest_cache=cache)
            return TaskOutcome(task.task_id,'failed',diagnostic=str(error))

        def run(prepared):
            task,root,native,payload,names,seed=prepared
            kwargs={}
            if request.progress_reporter_factory is not None:
                kwargs['progress_reporter']=request.progress_reporter_factory(task.task_id)
            parameters=inspect.signature(self.executor.run).parameters
            kwargs={key:value for key,value in kwargs.items() if key in parameters}
            try:
                self.executor.run(root,native,**kwargs)
            except Exception as error:
                return retain_failure(prepared,error)
            terminal=(root/'native_protocol_complete.out').read_text().split()
            if len(terminal)!=2 or terminal[0]!='DICE_NATIVE_PROTOCOL_COMPLETE':
                raise ValueError('native protocol did not complete its command sequence')
            terminal_step=int(terminal[1])
            budget=payload['effective']['total_steps']
            if payload['observation_kind']=='md' and budget is not None and terminal_step!=budget:
                raise ValueError('native protocol did not execute its declared step budget')
            rows,_index=scan_protocol_dump(root/payload['trajectory'],type_map=payload['type_map'],seed=seed,
                variable_types=payload['variable_types'],variable_count=payload['variable_count'])
            files=['seed.xyz','conf.lmp','protocol.lammps','input.lammps','dice_task_setup.lammps',
                   'native_protocol_complete.out',payload['trajectory'],*names,*payload['inputs']]
            if payload['observation_kind'] == 'optimization':
                endpoints, _ = scan_protocol_dump(root/'protocol_endpoint.dump', type_map=payload['type_map'],
                    seed=seed, variable_types=payload['variable_types'], variable_count=payload['variable_count'])
                if len(endpoints) != 1 or endpoints[0]['native_step'] != terminal_step:
                    raise ValueError('native optimization endpoint is incomplete')
                if rows[-1]['native_step'] != terminal_step:
                    with (root/payload['trajectory']).open('ab') as output, (root/'protocol_endpoint.dump').open('rb') as source:
                        shutil.copyfileobj(source, output)
                    rows = (*rows, {**endpoints[0], 'ordinal': len(rows)})
                files.append('protocol_endpoint.dump')
            if len(names)>1:files.append('model_devi.out')
            result=dict(schema='dice.native-protocol-result',revision=2,status='completed',task_id=task.task_id,
                context_digest=_file_digest(root/'task_context.json',cache=cache),frames=rows,terminal_native_step=terminal_step,
                files={name:_file_digest(root/name,cache=cache) for name in files})
            (root/'protocol_result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
            return TaskOutcome(task.task_id,'completed')
        outcomes=execute_lammps_tasks(prepared,executor=self.executor,sample_dir=evidence.sample_dir,
            task_id=lambda item:item[0].task_id,execute=run)
        return SamplingExecutionResult(evidence.sample_dir,len(tasks),sum(outcome.status=='failed' for outcome in outcomes))
