"""Run the reviewed evaluation set through the complete production answer path.

Checks are mechanical: abstention/clarification markers, required citations,
verbatim source facts for grounded questions, and expected filter counts.
Results feed the domain reviewer's worksheet; pass/fail is consistency with the
recorded expectations, not a clinical accuracy score.

In-process (uses .env backend selection):  python evaluate_reviewed.py
Against a running API:                     python evaluate_reviewed.py --url http://127.0.0.1:8001
Model-free subset only (no LLM required):  --grounded-only
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

SET_PATH = Path('evaluation/questions.json')
ABSTAIN_MARKERS = ('not available in drugcentral', 'did you mean', 'please confirm')


def check(entry, answer):
    """Return a list of mechanical check failures for one answer."""
    expect, low = entry['expect'], answer.lower()
    failures = []
    behavior = expect['behavior']
    if behavior == 'abstain':
        if not any(marker in low for marker in ABSTAIN_MARKERS):
            failures.append('expected abstention: ' + expect.get('abstain_reason', ''))
    elif behavior == 'clarify':
        if 'multiple' not in low or 'which' not in low:
            failures.append('expected a clarification request listing the candidates')
    else:
        for required in expect.get('must_contain', []):
            if required.lower() not in low:
                failures.append(f'missing expected content: {required[:80]}')
        for forbidden in expect.get('must_not_contain', []):
            if forbidden.lower() in low:
                failures.append(f'forbidden content present: {forbidden[:80]}')
    if behavior != 'abstain':
        for url in expect.get('cite', []):
            if url and url not in answer:
                failures.append(f'missing citation: {url}')
    return failures


def percentile(ordered, fraction):
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default=str(SET_PATH))
    parser.add_argument('--url', help='HTTP mode against a running API; default is in-process')
    parser.add_argument('--grounded-only', action='store_true', help='Skip entries that need the LLM')
    parser.add_argument('--output', default='benchmarks/reviewed-evaluation.json')
    parser.add_argument('--worksheet', default='evaluation/review_worksheet.json')
    args = parser.parse_args()
    questions = json.loads(Path(args.set).read_text())
    entries = questions['questions']
    if args.grounded_only:
        entries = [e for e in entries if e['expect'].get('route') == 'grounded']
    ask = None
    if args.url:
        import httpx
        client = httpx.Client(timeout=180, trust_env=False)
        ask = lambda q: client.post(args.url.rstrip('/') + '/api/chat', json={'question': q}).json()['answer']
    else:
        from agent import agent_executor, invoke_english_only
        ask = lambda q: invoke_english_only(agent_executor, q)
    results = []
    for number, entry in enumerate(entries, 1):
        started = time.perf_counter()
        error = None
        try:
            answer = ask(entry['question'])
        except Exception as exc:
            answer, error = '', type(exc).__name__
        seconds = time.perf_counter() - started
        failures = check(entry, answer) if error is None else [f'error: {error}']
        results.append({'id': entry['id'], 'category': entry['category'],
                        'question': entry['question'], 'seconds': round(seconds, 3),
                        'failures': failures, 'answer': answer})
        if number % 20 == 0:
            print(f'Evaluated {number}/{len(entries)}', flush=True)
    summary = {}
    for category in sorted({r['category'] for r in results}):
        rows = [r for r in results if r['category'] == category]
        latency = sorted(r['seconds'] for r in rows)
        summary[category] = {'cases': len(rows),
                             'passed': sum(not r['failures'] for r in rows),
                             'failed': sum(bool(r['failures']) for r in rows),
                             'p50_seconds': statistics.median(latency),
                             'p95_seconds': percentile(latency, 0.95)}
    report = {'set': args.set, 'dataset_versions': questions.get('dataset_versions'),
              'url': args.url, 'grounded_only': args.grounded_only,
              'scope': 'Mechanical expectation checks, not clinical accuracy.',
              'summary': summary,
              'passed': sum(not r['failures'] for r in results), 'total': len(results),
              'results': [{k: v for k, v in r.items() if k != 'answer'} for r in results]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    worksheet = [{'id': e['id'], 'question': e['question'], 'expect': e['expect'],
                  'answer': r['answer'], 'failures': r['failures'], 'review': e['review']}
                 for e, r in zip(entries, results)]
    Path(args.worksheet).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worksheet).write_text(json.dumps({'instructions': 'Domain reviewer: set '
        'review.status to approved/rejected with reviewer name and decision notes.',
        'cases': worksheet}, indent=2))
    print(json.dumps({'passed': report['passed'], 'total': report['total'],
                      'by_category': {k: v for k, v in summary.items()}}, indent=2))
    if report['passed'] != report['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
