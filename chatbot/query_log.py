"""Append-only per-query JSONL log for failure analysis and benchmarking.

Set QUERY_LOG_PATH to an empty value to disable. The uvicorn request log and
the metrics store remain question-free; this file is the single explicit place
where question text is recorded, and writing it never affects request handling.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chatbot.failure_reasons import categorize

VERIFIED_ROUTES = {'vector-rag', 'direct', 'direct-reverse', 'direct-indications',
                   'direct-comparison', 'direct-label', 'graph'}


def _query_context(observations):
    for obs in reversed(list(observations or ())):
        if isinstance(obs, dict) and isinstance(obs.get('query_context'), dict):
            return obs['query_context']
    return {}


def build_record(question, trace, seconds, intent=None):
    """Assemble the structured log record from a captured answer trace."""
    route = trace.get('route')
    reason = trace.get('reason')
    context = _query_context(trace.get('observations'))
    return {
        'ts': datetime.now(timezone.utc).isoformat(),
        'question': question,
        'intent': context.get('intent') or intent,
        'drugs': context.get('drugs', []),
        'filters': context.get('filters', []),
        'retrieved_fields': context.get('fields', {}),
        'route': route,
        'verified': route in VERIFIED_ROUTES,
        'reason_category': categorize(route, reason, trace.get('evidence'), trace.get('observations')),
        'reason': reason,
        'seconds': round(seconds, 3),
    }


def log_query(record):
    path = os.getenv('QUERY_LOG_PATH', 'logs/query.jsonl')
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a') as handle:
            handle.write(json.dumps(record, default=str) + '\n')
    except OSError:
        pass  # Logging must never break a request.
