"""Linux child exec guard: share worker process group and die with its parent."""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import sys

from workflow_backends.lammps.execution_mode import command_for_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--mode', choices=('ordinary', 'kokkos'), required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        raise ValueError('native GCMC/MD child ownership currently requires Linux')
    executable = Path(args.executable)
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('native child requires the resolved registered LAMMPS executable')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or visible == '-1':
        raise ValueError('native LAMMPS segment requires an explicit GPU assignment')
    # A tiny fresh interpreter sets the guard before exec. Never run ctypes or
    # Python preexec_fn after fork from a Torch evaluator with live threads.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot establish native parent-death guard')
    if os.getppid() != args.parent:
        raise RuntimeError('owning GCMC worker exited before native launch')
    argv = (str(executable), *command_for_mode(args.mode)[1:])
    os.execv(str(executable), argv)


if __name__ == '__main__':
    main()
