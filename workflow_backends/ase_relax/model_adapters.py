"""Explicit native model composition; imports happen only in bound children."""
import os


class DeepMDTF:
    policy_identity = {'name': 'committee-force-deviation', 'revision': 1}

    def initialize(self):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_NUM_INTRAOP_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'

    def preflight(self, models):
        from deepmd.calculator import DP
        if not models:
            raise ValueError('DeepMD relax requires ordered models')

    def create(self, model):
        from deepmd.calculator import DP
        return DP(model=model)

    @staticmethod
    def probe_code():
        return ('import tensorflow as tf\n'
                'assert tf.config.list_physical_devices("GPU")\n'
                'tf.config.set_soft_device_placement(False)\n'
                'with tf.device("/GPU:0"):\n'
                '    y=tf.reduce_sum(tf.ones((16,))**2)\n'
                'assert "GPU:0" in y.device and float(y.numpy())==16\n')

    def close(self):
        pass


class DeepMDPT(DeepMDTF):
    def initialize(self):
        os.environ['DP_INTRA_OP_PARALLELISM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['DP_INTER_OP_PARALLELISM_THREADS'] = '1'

    @staticmethod
    def probe_code():
        return ('import torch; x=torch.ones(16, device="cuda:0"); '
                'y=x*x; torch.cuda.synchronize(); assert y.is_cuda and y.sum().item()==16')


def model_adapter(family):
    adapters = {'deepmd-tf': DeepMDTF, 'deepmd-pt': DeepMDPT}
    try:
        return adapters[family]()
    except KeyError as error:
        raise ValueError('persistent relax model capability is unsupported') from error
