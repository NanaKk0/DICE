"""Compile DICE's ordered programs from a normal single-stage MD protocol."""
from workflow_backends.lammps.protocol import command_spans


def native_cell_control(text):
    if '# DICE_CAPABILITY: nvt_cell_program' in text:
        return None
    commands=command_spans(text.splitlines(keepends=True))
    fixes=[row for _,_,row in commands if row[0]=='fix' and len(row)>3 and row[3] in {'nvt','npt','nve','langevin'}]
    if len(fixes)!=1 or fixes[0][3:5]!=['nvt','temp'] or len(fixes[0])<8:
        raise ValueError('nvt_cell_program needs one native NVT thermostat')
    row=fixes[0]
    return {'group':row[2],'damping':row[7],'extra':row[8:]}


def prepare_sampling_protocol(text, kind):
    if kind=='md_mc':
        lines=text.splitlines(keepends=True);commands=command_spans(lines)
        if any(row==['include','dice_md_mc.lammps'] for _,_,row in commands):
            return text if '# DICE_CAPABILITY: md_mc' in text else '# DICE_CAPABILITY: md_mc\n'+text
        fixes=[item for item in commands if item[2][0]=='fix' and len(item[2])>3 and item[2][3] in {'nvt','npt'}]
        runs=[item for item in commands if item[2][0]=='run']
        if len(fixes)!=1 or len(runs)!=1:
            raise ValueError('md_mc needs one native NVT/NPT thermostat and one run')
        start,end,row=fixes[0];row=row.copy()
        if row[4:5]!=['temp'] or len(row)<8:raise ValueError('md_mc native thermostat syntax is unsupported')
        row[5:7]=['${dice_temperature}']*2
        if row[3]=='npt':
            for i in range(8,len(row)-3):
                if row[i] in {'iso','aniso','tri','x','y','z','xy','xz','yz'}:row[i+1:i+3]=['${dice_pressure}']*2
            if 'flip' not in row:row+=['flip','no']
        edits={start:(end,' '.join(row)+'\n'),runs[0][0]:(runs[0][1],'include dice_md_mc.lammps\n')}
        for start,(end,replacement) in sorted(edits.items(),reverse=True):lines[start:end]=[replacement]
        cadence=next((r[4] for _,_,r in commands if r[0]=='dump' and len(r)>4 and r[4].isdigit()),'20')
        declaration='' if any(r[:2]==['variable','dice_trajectory_every_steps'] for _,_,r in commands) else f'variable dice_trajectory_every_steps equal {cadence}\n'
        return '# DICE_CAPABILITY: md_mc\n'+declaration+''.join(lines)
    if kind=='nvt_cell_program':
        control=native_cell_control(text)
        if control is None:return text
        lines=text.splitlines(keepends=True)
        for start,end,row in reversed(command_spans(lines)):
            if row[0]=='run' or (row[0]=='fix' and row[3]=='nvt'):
                lines[start:end]=[]
        return '# DICE_CAPABILITY: nvt_cell_program\nvariable dice_trajectory_every_steps equal 20\n'+''.join(lines)
    if kind not in {'nvt_program','npt_program'}:
        return text
    lines=text.splitlines(keepends=True)
    commands=command_spans(lines)
    if any(row==['label','dice_segment_loop'] for _,_,row in commands):
        return text  # Historical authored schedule remains readable.
    ensemble=kind.removesuffix('_program')
    fixes=[item for item in commands if item[2][0]=='fix' and len(item[2])>3 and item[2][3] in {'nvt','npt','langevin'}]
    runs=[item for item in commands if item[2][0]=='run']
    if len(fixes)!=1 or len(runs)!=1 or fixes[0][2][3]!=ensemble:
        raise ValueError(f'{kind} needs one matching native thermostat and run; complete native programs can use recipe-only')
    start,end,row=fixes[0];run_start,run_end,_run=runs[0]
    if end>run_start or row[4:5]!=['temp']:
        raise ValueError(f'{kind} needs its thermostat before the native run')
    row=row.copy();row[5:7]=['${dice_tstart}','${dice_tstop}']
    if ensemble=='npt':
        for i in range(8,len(row)-3):
            if row[i] in {'iso','aniso','tri','x','y','z','xy','xz','yz'}:
                row[i+1:i+3]=['${dice_pressure}','${dice_pressure}']
    identifier=row[1]
    tail=lines[run_end:]
    for a,b,tokens in reversed(command_spans(tail)):
        if tokens==['unfix',identifier]:tail[a:b]=[]
    cadence=next((tokens[4] for _,_,tokens in commands if tokens[0]=='dump' and len(tokens)>4 and tokens[4].isdigit()),'20')
    declaration='' if any(tokens[:2]==['variable','dice_trajectory_every_steps'] for _,_,tokens in commands) else f'variable dice_trajectory_every_steps equal {cadence}\n'
    loop=('variable dice_tstart file dice_temperature_start.dat\n'
          'variable dice_tstop file dice_temperature_stop.dat\n'
          'variable dice_segment_steps file dice_temperature_steps.dat\n'
          'label dice_segment_loop\n'+' '.join(row)+'\n'+''.join(lines[end:run_start])+
          'run ${dice_segment_steps}\n'+f'unfix {identifier}\n'+
          'next dice_tstart dice_tstop dice_segment_steps\njump SELF dice_segment_loop\n')
    return declaration+''.join(lines[:start])+loop+''.join(tail)
