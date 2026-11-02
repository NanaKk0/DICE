"""Narrow native evidence classifier, independent of scientific task failure."""
from contextvars import ContextVar


def classify_device_fault(detail: str) -> str | None:
    text = detail.lower()
    if 'mps server is not ready' in text:
        return 'mps_not_ready'
    if 'fatal gpu error' in text:
        return 'mps_fatal'
    if 'illegal memory access' in text or 'cudaerrorillegaladdress' in text:
        return 'cuda_illegal_address'
    return None


fault_reporter = ContextVar('dice_device_fault_reporter', default=None)


def report_device_fault(detail: str) -> bool:
    kind = classify_device_fault(detail)
    report = fault_reporter.get()
    if kind is not None and report is not None:
        report(kind, detail[-4096:])
    return kind is not None
