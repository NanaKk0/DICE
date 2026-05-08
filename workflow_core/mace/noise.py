from __future__ import annotations

import io
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout


MACE_NOISE_SUBSTRINGS = [
    "Cuequivariance acceleration will be disabled.",
    "_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))",
]

MACE_NOISE_PATTERNS = [
    re.compile(r".*weights_only=False.*", re.IGNORECASE),
]


class _FilteringTextStream(io.TextIOBase):
    def __init__(self, underlying, substrings: list[str], patterns: list[re.Pattern[str]]):
        self._underlying = underlying
        self._substrings = substrings
        self._patterns = patterns
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        self._underlying.flush()

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line + "\n")
        return len(data)

    def _emit(self, line: str) -> None:
        if any(text in line for text in self._substrings):
            return
        if any(pattern.search(line) for pattern in self._patterns):
            return
        self._underlying.write(line)


@contextmanager
def suppress_known_mace_noise():
    filtered_stdout = _FilteringTextStream(sys.stdout, MACE_NOISE_SUBSTRINGS, MACE_NOISE_PATTERNS)
    filtered_stderr = _FilteringTextStream(sys.stderr, MACE_NOISE_SUBSTRINGS, MACE_NOISE_PATTERNS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=UserWarning)
        with redirect_stdout(filtered_stdout), redirect_stderr(filtered_stderr):
            try:
                yield
            finally:
                filtered_stdout.flush()
                filtered_stderr.flush()
