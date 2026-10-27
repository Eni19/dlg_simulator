"""Request-scoped telemetry; command-line scientific routines need no subscriber."""
from contextvars import ContextVar
import math

reporter = ContextVar("experiment_progress_reporter", default=None)


def report(phase, text, current=None, total=None, loss=None):
    callback = reporter.get()
    if callback is None:
        return
    event = {"type": "progress", "phase": phase, "text": text,
             "current": current, "total": total}
    if loss is not None and math.isfinite(loss):
        event["loss"] = loss
    callback(event)
