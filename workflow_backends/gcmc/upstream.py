"""Fail-closed qualification of the installed, unmodified official MC kernel."""
import hashlib
import importlib.util
from pathlib import Path


COMMIT = 'd9caf0f83cf20db61bdc283777b33c57edd1f8bf'
KERNEL_SHA256 = {
    'gcmc.py': 'cfb3fa246f820c3579aad526186e34072564e146dbccb3e8aae0815dd3a1c12f',
    'base_simulator.py': '1bff4dc4accb905c0bc44b1d5e0b60057276c4314edcd657532ca8d8090aef7d',
    'operations.py': '0a66bedd82539272ddb2f95a706e1bcfaaac419ad00c1caf77b774b143baab9a',
    'eos.py': 'd7534e08763cdc7e2424b9b694e1d3abd11eedbde043a8cc06bf0aa29cc80eca',
    'utilities.py': '8c1fbe433f845c6add895b2ecfc3ee90d801430cdecd0cdf8c533783249b86af',
    'version.py': '20c8173adfdfeb7d999cfbdb12ba7f0ed3423432b5fff31fcfcfb62d62216540',
}


def load_official():
    spec = importlib.util.find_spec('flames')
    if spec is None or not spec.origin:
        raise ValueError(f'install official FLAMES {COMMIT} in the selected sampling Python environment')
    root = Path(spec.origin).resolve().parent
    for name, expected in KERNEL_SHA256.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'unqualified FLAMES kernel: {name}; expected official {COMMIT}')
    from flames.gcmc import GCMC
    return GCMC
