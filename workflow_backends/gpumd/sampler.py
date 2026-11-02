from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
import threading
import time
from typing import Callable, Protocol, Sequence
from ase import Atoms
from workflow_backends.common.model_files import link_model_file
from workflow_backends.common.sampling_pool import execute_sampling_tasks
from workflow_backends.sample_artifacts import (
    ExplicitSampleTaskContext,
    SampleTaskContext,
    canonical_selected_seed,
    explicit_task_seed,
    publish_selected_seeds,
    write_task_context,
)
from workflow_core.structures.formats import write_extxyz_structures
from workflow_backends.gpumd.observation_source import GpumdObservationSource
from workflow_artifacts.lifecycle.sample_conditions import (
    SamplingPlan,
    SystemSamplingCondition,
)
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_runtime.lifecycle.progress import (
    OperationProgressReporter,
    SamplingSnapshot,
    SamplingSystemProgress,
)

class GpumdExecutor(Protocol):
    def run(self,task_dir:Path,task:dict)->object:...


@dataclass(frozen=True, slots=True)
class GpumdTask:
    task_index: int
    task_dir: Path
    payload: dict


@dataclass(slots=True)
class GpumdSampler:
    run_json_path:Path; iter_index:int; round_index:int; executor:GpumdExecutor
    def sample(self,*,recipe:str|Path|None=None,group_recipes:dict[str,str|Path]|None=None,sample_dir:str|Path,models:Sequence[str|Path],systems:dict[str,list[Atoms]],steps:int|None,temperatures_k:Sequence[float]|None=None,ensemble:str|None=None,pressure_bar:float|None=None,plan:SamplingPlan|ExplicitSamplingPlan|None=None,plumed_sources:dict[str,str|Path],progress_callback:Callable[[SamplingSnapshot],None]|None=None,progress_reporter_factory:Callable[[str],OperationProgressReporter]|None=None)->None:
        if plumed_sources: raise ValueError("GPUMD lifecycle rejects PLUMED")
        effective_plan = _effective_plan(
            plan=plan,
            systems=systems,
            temperatures_k=temperatures_k,
            ensemble=ensemble,
            pressure_bar=pressure_bar,
            round_index=self.round_index,
        )
        explicit=isinstance(effective_plan,ExplicitSamplingPlan)
        all_tasks=effective_plan.tasks_for_round(self.round_index) if explicit else effective_plan.tasks
        indexed_tasks=tuple((index,task) for index,task in enumerate(all_tasks) if not explicit or task.kind!="neb")
        if not indexed_tasks: raise ValueError("GPUMD sampling requires at least one active MD task")
        if explicit and any(task.kind not in {"nvt", "npt"} for _,task in indexed_tasks):
            raise ValueError("GPUMD supports only nvt and npt MD tasks")
        if explicit:
            if recipe is not None or group_recipes is None: raise ValueError("explicit GPUMD sampling requires group recipes")
            sources={str(key):Path(value) for key,value in group_recipes.items()}
            active_groups={task.condition_group_id for _,task in indexed_tasks}
            if not active_groups<=set(sources): raise ValueError("GPUMD group recipes must cover active MD groups")
        else:
            if recipe is None or group_recipes is not None: raise ValueError("legacy GPUMD sampling requires one recipe")
            sources={"legacy":Path(recipe)}
        for source in dict.fromkeys(sources.values()): self._validate(source)
        model_paths=_canonical_nep_model_paths(models)
        selected_seeds={
            system: canonical_selected_seed(seeds[0 if explicit else self._seed(system)%len(seeds)],system_id=system)
            for system,seeds in systems.items()
        }
        root=Path(sample_dir)
        publish_selected_seeds(root/"seeds.xyz",tuple(selected_seeds.values()))
        total_tasks=len(indexed_tasks)
        active_system_ids=tuple(system for system in selected_seeds if any(task.system_id==system for _,task in indexed_tasks))
        task_counts={system:sum(task.system_id==system for _,task in indexed_tasks) for system in active_system_ids}
        ensembles=tuple(dict.fromkeys(task.ensemble for _,task in indexed_tasks))
        completed_by_system={system:0 for system in active_system_ids}
        started_at=time.monotonic()
        progress_lock=threading.RLock()

        def progress(*,active_task_id=None,active_system_id=None,active_temperature_k=None,active_pressure_bar=None):
            completed=sum(completed_by_system.values())
            elapsed=time.monotonic()-started_at
            eta=None if completed==0 or completed>=total_tasks else elapsed*(total_tasks-completed)/completed
            return SamplingSnapshot(
                backend="GPUMD",
                model_shape="single" if len(model_paths)==1 else "committee",
                completed_tasks=completed,
                total_tasks=total_tasks,
                systems=len(active_system_ids),
                temperatures=len(indexed_tasks),
                ensemble=ensembles[0].upper() if len(ensembles)==1 else "MIXED",
                pressure_bar=active_pressure_bar,
                steps_per_task=max(int(task.steps if explicit else steps) for _,task in indexed_tasks),
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                rows=(
                    SamplingSystemProgress("ALL",completed,total_tasks),
                    *(SamplingSystemProgress(system,completed_by_system[system],task_counts[system]) for system in active_system_ids),
                ),
                active_task_id=active_task_id,
                active_system_id=active_system_id,
                active_temperature_k=active_temperature_k,
            )

        if progress_callback is not None: progress_callback(progress())
        prepared_tasks: list[GpumdTask] = []
        for task_index, condition in indexed_tasks:
            system=condition.system_id; seed=selected_seeds[system]; temperature=condition.temperature_k; pressure=condition.pressure_bar
            source=sources[condition.condition_group_id if explicit else "legacy"]
            cadence=self._cadence(source)
            task_steps=int(condition.steps if explicit else steps)
            task_dir=root/"tasks"/f"task_{task_index:06d}"; task_dir.mkdir(parents=True,exist_ok=True)
            self._render(task_dir,source,model_paths,seed,task_steps,temperature,pressure,cadence)
            if explicit:
                presentations=tuple(sorted(task_dir.glob("nep_*.txt")))
                write_task_context(task_dir/"task_context.json",ExplicitSampleTaskContext(schema="dice.lifecycle.sample-task-context",schema_revision=2,task_index=task_index,task_id=condition.task_id,condition_group_id=condition.condition_group_id,kind=condition.kind,sampling_role_id=condition.sampling_role_id,selection_group_id=condition.selection_group_id,system_id=condition.selection_group_id,source_frame_index=condition.source_frame_index,selected_structure_id=str(seed.info["structure_id"]),temperature_k=float(temperature),pressure_bar=pressure,total_steps=task_steps,trajectory_every_steps=cadence,native_timestep=float(self._time_step(source)),recipe_binding=condition.recipe_binding,recipe_sha256=self._digest(source),rendered_input_sha256=self._digest(task_dir/"run.in"),setup_sha256=self._digest(task_dir/"model.xyz"),native_seed=explicit_task_seed(iter_index=self.iter_index,round_index=self.round_index,sampling_role_id=condition.sampling_role_id),ordered_models=tuple(path.name for path in presentations),model_digests=tuple(self._digest(path) for path in presentations)))
            else:
                write_task_context(task_dir/"task_context.json",SampleTaskContext.create(task_index=task_index,system_id=system,ensemble=condition.ensemble,temperature_k=float(temperature),pressure_bar=pressure,steps=task_steps,cadence=cadence,selected_seed_id=str(seed.info["structure_id"])))
            prepared_tasks.append(
                GpumdTask(
                    task_index,
                    task_dir,
                    {"cadence":cadence,"steps":task_steps,"thermo_cadence":self._thermo_cadence(source),"system_id":system,"ensemble":condition.ensemble,"temperature_k":temperature,"pressure_bar":pressure},
                )
            )

        def execute_one(prepared: GpumdTask) -> GpumdTask:
            task_id=f"task_{prepared.task_index:06d}"
            system=str(prepared.payload["system_id"])
            temperature=float(prepared.payload["temperature_k"])
            pressure=prepared.payload["pressure_bar"]
            with progress_lock:
                active=progress(active_task_id=task_id,active_system_id=system,active_temperature_k=temperature,active_pressure_bar=pressure)
                if progress_callback is not None: progress_callback(active)
            reporter=None if progress_reporter_factory is None else progress_reporter_factory(task_id)
            try:
                _run_with_optional_progress(
                    self.executor.run,
                    prepared.task_dir,
                    prepared.payload,
                    progress_reporter=reporter,
                    sampling_snapshot=active,
                )
            finally:
                if reporter is not None and not reporter.closed: reporter.close()
            if any(not (prepared.task_dir/name).is_file() or (prepared.task_dir/name).stat().st_size<=0 for name in ("active.out","active.xyz")): raise RuntimeError("missing active.xyz or active.out")
            with progress_lock:
                completed_by_system[system]+=1
                if progress_callback is not None: progress_callback(progress())
            return prepared

        try:
            execute_sampling_tasks(
                tuple(prepared_tasks),
                context=getattr(self.executor, "context", None),
                task_id=lambda prepared: f"task_{prepared.task_index:06d}",
                execute=execute_one,
            )
        except Exception as exc:
            raise RuntimeError(
                f"GPUMD lifecycle sample block failed atomically: {exc}"
            ) from exc
    def _validate(self,path:Path)->None:
        text=path.read_text(encoding="utf-8"); lines=[line.split("#",1)[0].strip().split() for line in text.splitlines()]; heads=[item[0].lower() for item in lines if item]
        if heads.count("velocity")!=1 or heads.count("ensemble")!=1 or heads.count("run")!=1 or "potential" in heads or "active" in heads: raise ValueError("GPUMD lifecycle recipe requires exactly one velocity, ensemble, and run; no potential or active")
    def _render(self,task:Path,source:Path,models:Sequence[str|Path],seed:Atoms,steps:int,temp:float,pressure:float|None,cadence:int)->None:
        write_extxyz_structures(task / "model.xyz", [seed], allow_empty=False)
        names=[]
        for i,model in enumerate(models): target=task/f"nep_{i}.txt"; link_model_file(model,target); names.append(target.name)
        lines=[]
        for line in source.read_text(encoding="utf-8").splitlines():
            head=line.split()[0].lower() if line.split() else ""
            if head=="velocity": lines.append(f"velocity {temp:g}")
            elif head=="ensemble": lines.append(f"ensemble nvt_ber {temp:g} {temp:g} 100" if pressure is None else f"ensemble npt_ber {temp:g} {temp:g} 100 {pressure}")
            elif head=="run": lines.append(f"run {steps}")
            elif head=="dump_position": continue
            else: lines.append(line)
        lines=[*(f"potential {name}" for name in names),f"dump_position {cadence}",f"active {cadence} 1 1 1 0.0",*lines]; (task/"run.in").write_text("\n".join(lines)+"\n",encoding="utf-8")
    def _cadence(self,path:Path)->int:
        match=re.search(r"(?m)^\s*dump_position\s+(\d+)",path.read_text(encoding="utf-8")); return int(match.group(1)) if match else 20
    def _thermo_cadence(self,path:Path)->int|None:
        match=re.search(r"(?m)^\s*dump_thermo\s+(\d+)\s*(?:#.*)?$",path.read_text(encoding="utf-8")); return int(match.group(1)) if match else None
    def _seed(self,system:str)->int:return int.from_bytes(hashlib.sha256(f"{self.run_json_path.resolve()}\0{self.iter_index}\0{self.round_index}\0{system}".encode()).digest()[:4],"big")
    def _time_step(self,path:Path)->float:
        match=re.search(r"(?m)^\s*time_step\s+([^\s#]+)",path.read_text(encoding="utf-8")); return float(match.group(1)) if match else 1.0
    def _digest(self,path:Path)->str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_nep_model_paths(models: Sequence[str | Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for model in models:
        path = Path(model)
        if path.is_symlink() or path.suffix != ".txt" or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"GPUMD lifecycle requires a non-empty regular native .txt model: {path}")
        paths.append(path.resolve())
    if not paths:
        raise ValueError("GPUMD lifecycle requires at least one native .txt model")
    return tuple(paths)


def _effective_plan(
    *,
    plan: SamplingPlan | ExplicitSamplingPlan | None,
    systems: dict[str, list[Atoms]],
    temperatures_k: Sequence[float] | None,
    ensemble: str | None,
    pressure_bar: float | None,
    round_index: int,
) -> SamplingPlan | ExplicitSamplingPlan:
    if plan is None:
        if ensemble not in {"nvt", "npt"} or not temperatures_k:
            raise ValueError("GPUMD lifecycle requires a resolved sampling plan")
        if ensemble == "npt" and pressure_bar is None:
            raise ValueError("GPUMD npt requires pressure")
        plan = SamplingPlan.create(tuple(
            SystemSamplingCondition(
                system_id,
                ensemble,
                tuple(float(value) for value in temperatures_k),
                () if ensemble == "nvt" else (float(pressure_bar),),
            )
            for system_id in systems
        ))
    plan_ids=(tuple(role for role,_group in plan.selection_plan_for_round(round_index).role_to_group) if isinstance(plan,ExplicitSamplingPlan) else tuple(system.system_id for system in plan.systems))
    if plan_ids != tuple(systems):
        raise ValueError("GPUMD resolved sampling plan must exactly match configured system order")
    return plan


def _run_with_optional_progress(operation, task_dir: Path, task: dict, **kwargs):
    parameters=inspect.signature(operation).parameters
    for name in tuple(kwargs):
        if name not in parameters: kwargs.pop(name)
    return operation(task_dir,task,**kwargs)
