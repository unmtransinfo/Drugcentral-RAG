"""Request-local evidence capture; no effect when evaluation tracing is disabled."""
from contextlib import contextmanager
from contextvars import ContextVar
import json

_active = ContextVar('evaluation_trace', default=None)

@contextmanager
def capture_trace():
    trace = {'recorded': False, 'route': 'unknown', 'reason': None, 'evidence': [], 'observations': []}
    token = _active.set(trace)
    try:
        yield trace
    finally:
        _active.reset(token)


def record_trace(route, evidence=(), observations=(), reason=None):
    """Record the selected route; `reason` distinguishes abstain exits."""
    trace = _active.get()
    if trace is None:
        return
    # Replace earlier attempts: only evidence used by the selected answer counts.
    decoded = []
    for item in observations:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except (ValueError, TypeError):
                pass
        decoded.append(item)
    trace.update(recorded=True, route=route, reason=reason, evidence=list(evidence), observations=decoded)
