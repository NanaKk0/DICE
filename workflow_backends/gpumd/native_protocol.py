"""GPUMD control binding retaining native method, pressure axes and coupling."""
from workflow_core.common.native_protocol import NativeProtocolRequest


def compile_gpumd_protocol(text: str, request: NativeProtocolRequest) -> str:
    lines=text.splitlines(keepends=True)
    ensembles=[(i,line.split('#',1)[0].split()) for i,line in enumerate(lines)
               if line.split('#',1)[0].split()[:1]==['ensemble']]
    runs=[(i,line.split('#',1)[0].split()) for i,line in enumerate(lines)
          if line.split('#',1)[0].split()[:1]==['run']]
    edits={};prefix=''
    if request.temperature_k is not None or request.pressure_bar is not None or request.ensemble is not None:
        if len(ensembles)!=1:
            raise ValueError('GPUMD control override requires one native ensemble')
        i,row=ensembles[0];method=row[1]
        ensemble='npt' if method.startswith('npt_') else 'nvt' if method.startswith('nvt_') else method
        if request.ensemble is not None and request.ensemble!=ensemble:
            if request.ensemble=='nvt' and method=='npt_ber':
                row=row[:5];row[1]='nvt_ber'
            else:
                raise ValueError('GPUMD ensemble switch needs a matching native protocol with its coupling parameters')
        if request.temperature_k is not None:
            if len(row)<5 or not row[1].startswith(('nvt_','npt_')):
                raise ValueError('GPUMD temperature override needs native thermostat targets')
            row[2:4]=[format(request.temperature_k,'.17g')]*2
        if request.pressure_bar is not None:
            # GPUMD pressure and elastic moduli use GPa: one bar = 1e-4 GPa.
            if not row[1].startswith('npt_') or len(row)-5 not in {3,7,13}:
                raise ValueError('GPUMD pressure override needs native NPT axes, elastic moduli and coupling')
            axes={3:1,7:3,13:6}[len(row)-5]
            for index in range(min(axes,3)):
                row[5+index]=format(request.pressure_bar*1e-4,'.17g')
        edits[i]=' '.join(row)+'\n'
    if request.steps is not None:
        if request.run is None and len(runs)!=1:
            raise ValueError('GPUMD steps override needs one run or targets.run')
        index=(request.run or 1)-1
        if not 0<=index<len(runs):raise ValueError('GPUMD targets.run is outside the native run inventory')
        i,row=runs[index];row[1]=str(request.steps);edits[i]=' '.join(row)+'\n'
    if request.timestep_ps is not None:
        entries=[(i,line.split('#',1)[0].split()) for i,line in enumerate(lines)
                 if line.split('#',1)[0].split()[:1]==['time_step']]
        if len(entries)>1:raise ValueError('GPUMD time_step override is ambiguous across stages')
        if entries:
            i,row=entries[0];row[1]=format(request.timestep_ps*1000,'.17g');edits[i]=' '.join(row)+'\n'
        else:prefix=f'time_step {request.timestep_ps*1000:.17g}\n'
    for i,value in edits.items():lines[i]=value
    return prefix+''.join(lines)
