"""Observable ordering and bounded I/O at the sampling-to-screen seam."""
from types import SimpleNamespace
from pathlib import Path
import builtins
import pytest

from ase import Atoms

from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
from workflow_backends.observation_source import (
    InMemoryObservationSource, ObservationLocator, ObservationRecord,
)


def test_mixed_replay_batches_each_owner_and_preserves_requested_order():
    class Store(InMemoryObservationSource):
        def __init__(self, task):
            rows, frames = [], []
            for index in range(3):
                identity = f"observation-{task}-{index}"
                frame = Atoms('H', positions=[[task, index, 0]])
                frame.info['structure_id'] = identity
                frames.append(frame)
                rows.append(ObservationRecord(identity, 'joint', task, index, index,
                    None, ObservationLocator('test', task, index, index)))
            super().__init__(rows, frames)
            self.read_requests = []

        def replay(self, identities, *, chunk_size):
            self.read_requests.append(tuple(identities))
            yield from super().replay(identities, chunk_size=chunk_size)

    md, neb = Store(0), Store(1)
    def branch(source):
        return SimpleNamespace(open_current=lambda _: SimpleNamespace(source=source))
    backend = HybridSamplingNativeOps(md=branch(md), neb=branch(neb))
    evidence = SimpleNamespace(round_index=0, plan=SimpleNamespace(
        tasks_for_round=lambda _: (SimpleNamespace(kind='nvt'), SimpleNamespace(kind='neb'))))
    source = backend.open_current(evidence).source
    requested = ('observation-0-2', 'observation-1-1', 'observation-0-0', 'observation-1-0')
    chunks = tuple(source.replay(requested, chunk_size=4))
    assert [[f.info['structure_id'] for f in chunk] for chunk in chunks] == [list(requested)]
    assert md.read_requests == [('observation-0-2', 'observation-0-0')]
    assert neb.read_requests == [('observation-1-1', 'observation-1-0')]


def test_lammps_sparse_replay_reads_selected_frames_and_retains_native_pbc(tmp_path, monkeypatch):
    from workflow_backends.lammps.observation_source import LammpsObservationSource
    from workflow_backends.sample_artifacts import SampleTaskContext

    task = tmp_path / 'tasks' / 'task_000000'
    task.mkdir(parents=True)
    trajectory = task / 'traj.lammpstrj'
    blocks = [
        f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\n'
        'ITEM: BOX BOUNDS pp pp ff\n0 4\n0 4\n0 4\n'
        f'ITEM: ATOMS id type x y z\n1 1 {step / 1000} 0 0\n'
        for step in range(0, 2001, 20)
    ]
    trajectory.write_text(''.join(blocks))
    context = SampleTaskContext.create(task_index=0, system_id='joint', ensemble='nvt',
        temperature_k=300, pressure_bar=None, steps=2000, cadence=20, selected_seed_id='seed')
    source = LammpsObservationSource(sample_dir=tmp_path, run_json_path=tmp_path/'run.json',
        iter_index=0, round_index=0, type_map=('H',), model_count=1, contexts=(context,))
    rows = source.catalog()
    read_bytes = 0
    original_open = Path.open
    original_builtin_open = builtins.open

    class Counted:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def __getattr__(self, name): return getattr(self.stream, name)
        def read(self, *args):
            nonlocal read_bytes
            value = self.stream.read(*args); read_bytes += len(value); return value
        def readlines(self, *args):
            nonlocal read_bytes
            value = self.stream.readlines(*args); read_bytes += sum(map(len, value)); return value
        def readline(self, *args):
            nonlocal read_bytes
            value = self.stream.readline(*args); read_bytes += len(value); return value
        def __iter__(self):
            while value := self.readline(): yield value

    def tracked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Counted(stream) if path == trajectory else stream
    monkeypatch.setattr(Path, 'open', tracked_open)
    def tracked_builtin_open(path, *args, **kwargs):
        stream = original_builtin_open(path, *args, **kwargs)
        return Counted(stream) if isinstance(path, (str, Path)) and Path(path) == trajectory else stream
    monkeypatch.setattr(builtins, 'open', tracked_builtin_open)
    wanted = (rows[-1], rows[0], rows[-2])
    frames = [f for chunk in source.replay(tuple(r.structure_id for r in wanted), chunk_size=2)
              for f in chunk]
    assert [f.info['structure_id'] for f in frames] == [r.structure_id for r in wanted]
    assert read_bytes < trajectory.stat().st_size // 4
    assert [f.pbc.tolist() for f in frames] == [[True, True, False]] * 3
    assert [f.positions[0, 0] for f in frames] == [2., .02, 1.98]
    trajectory.write_text(''.join(blocks).replace('1 1 2.0', '1 1 9.0'))
    with pytest.raises(ValueError, match='changed'):
        tuple(source.replay((rows[-1].structure_id,), chunk_size=1))


@pytest.mark.parametrize('backend',['relax','neb'])
def test_ase_observation_snapshot_rejects_geometry_mutation_after_catalog(tmp_path,backend):
    from ase.io import write
    from workflow_backends.ase_neb.adapter import _NebObservationSource
    from workflow_backends.ase_relax.adapter import _RelaxObservationSource
    task=tmp_path/'tasks/task_000000';task.mkdir(parents=True)
    frame=Atoms('H',positions=[[.1,0,0]],info={'structure_id':'one'})
    path=task/'observations.extxyz';write(path,frame,format='extxyz')
    rows=({'structure_id':'one','selection_group_id':'joint','task_index':0,'frame_index':0,
        'optimizer_step':2,'image_index':1,'force_deviation':0.,'sampling_role_id':'A',
        'condition_group_id':backend,'task_id':'task_000000'},)
    source=(_NebObservationSource if backend=='neb' else _RelaxObservationSource)(tmp_path,rows)
    row,=source.catalog()
    assert row.md_step is None
    assert row.optimizer_step == 2
    frame.positions[0,0]=.2
    replacement=path.with_suffix('.new');write(replacement,frame,format='extxyz');replacement.replace(path)
    with pytest.raises(ValueError,match='changed'):
        tuple(source.replay((row.structure_id,),chunk_size=1))


def test_gpumd_replay_preserves_native_pbc_and_rejects_replaced_snapshot(tmp_path):
    from ase.io import write
    from workflow_backends.gpumd.observation_source import GpumdObservationSource
    from workflow_backends.sample_artifacts import SampleTaskContext
    task=tmp_path/'tasks/task_000000';task.mkdir(parents=True)
    path=task/'active.xyz'
    frames=[Atoms('H',positions=[[i/10,0,0]],cell=[4]*3,pbc=[True,True,False]) for i in range(1,5)]
    write(path,frames,format='extxyz')
    (task/'active.out').write_text('1 .1\n2 .2\n3 .3\n4 .4\n')
    (task/'run.in').write_text('time_step 1\nrun 4\n')
    context=SampleTaskContext.create(task_index=0,system_id='joint',ensemble='nvt',
        temperature_k=300,pressure_bar=None,steps=4,cadence=1,selected_seed_id='seed')
    source=GpumdObservationSource(sample_dir=tmp_path,run_json_path=tmp_path/'run.json',
        iter_index=0,round_index=0,contexts=(context,))
    rows=source.catalog()
    actual=[f for chunk in source.replay((rows[-1].structure_id,rows[0].structure_id),chunk_size=1) for f in chunk]
    assert [f.positions[0,0] for f in actual]==[.4,.1]
    assert [f.pbc.tolist() for f in actual]==[[True,True,False]]*2
    replacement=path.with_suffix('.new');write(replacement,frames,format='extxyz');replacement.replace(path)
    with pytest.raises(ValueError,match='changed'):
        tuple(source.replay((rows[-1].structure_id,),chunk_size=1))
