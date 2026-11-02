"""LAMMPS model probes composed with shared bounded session recovery."""
import subprocess
from workflow_backends.lammps.mps import CudaMpsSession
from workflow_backends.common.managed_session import managed_device_session as _managed


def _probe(context, environment):
    codes = {
        'sampler.lammps.pt': ('import torch; x=torch.ones(16, device="cuda"); '
            'y=x*x; torch.cuda.synchronize(); assert y.sum().item()==16'),
        'sampler.lammps.tf': ('import tensorflow as tf\n'
            'assert tf.config.list_physical_devices("GPU")\n'
            'tf.config.set_soft_device_placement(False)\n'
            'with tf.device("/GPU:0"):\n'
            '    x=tf.ones((16,)); y=tf.reduce_sum(x*x)\n'
            'assert float(y.numpy())==16\n'),
    }
    environment['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    return subprocess.run([str(context.executable('python')), '-c', codes[context.role]],
        env=environment, text=True, capture_output=True, timeout=30, check=False)


def managed_device_session(context, policy, root, device, index):
    return _managed(context, policy, root, device, index,
                    session_factory=CudaMpsSession, probe_callback=_probe)
