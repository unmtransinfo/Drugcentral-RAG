"""Realistic-usage test of the complete HTTP API: latency, concurrency, memory.

Scenarios: direct-route latency (no LLM), queue saturation with slow agent
requests, concurrent session isolation, and memory/readiness probes. Limits are
asserted with --max-* options and the exit code reflects them; defaults are the
agreed starting gates, tune them with the team before scaling.

Requires a running API. Agent-route latency needs the model service; use
--agent-question to include it. Failures of the model service and database are
exercised separately by tests/test_api_auth.py mocks and by probing /api/ready while
the dependency is stopped.

Run: python load_test.py --url http://127.0.0.1:8001 --requests 40 --concurrency 6
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import statistics
import time
import uuid
from pathlib import Path
import httpx

DIRECT_QUESTION = 'What is metformin used for?'
FOLLOWUP = 'What is it used for?'


def percentile(ordered, fraction):
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0.0


def chat(client, url, question, session=None):
    payload = {'question': question}
    if session:
        payload['session_id'] = session
    started = time.perf_counter()
    try:
        response = client.post(url + '/api/chat', json=payload)
        return {'status': response.status_code, 'seconds': time.perf_counter() - started,
                'answer': response.json().get('answer', '') if response.status_code == 200 else ''}
    except httpx.HTTPError as exc:
        return {'status': 0, 'seconds': time.perf_counter() - started, 'answer': f'client error: {type(exc).__name__}'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--requests', type=int, default=40)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--agent-question', help='Also time a generated-answer question (uses the LLM)')
    parser.add_argument('--max-direct-p95', type=float, default=2.0)
    parser.add_argument('--max-error-rate', type=float, default=0.05)
    parser.add_argument('--max-agent-p95', type=float, default=120.0)
    parser.add_argument('--output', default='benchmarks/load-test.json')
    args = parser.parse_args()
    url = args.url.rstrip('/')
    report = {'url': url, 'limits': {'max_direct_p95': args.max_direct_p95,
              'max_error_rate': args.max_error_rate, 'max_agent_p95': args.max_agent_p95},
              'failures': []}
    with httpx.Client(timeout=180, trust_env=False) as client:
        ready = client.get(url + '/api/ready')
        report['ready_status'] = ready.status_code
        if ready.status_code != 200:
            raise SystemExit('API is not ready; aborting load test')
        before = client.get(url + '/api/metrics').json()

        # 1) Direct-route latency under concurrency (model-free path).
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            direct = list(pool.map(lambda _: chat(client, url, DIRECT_QUESTION), range(args.requests)))
        latency = sorted(r['seconds'] for r in direct)
        errors = sum(r['status'] != 200 for r in direct)
        report['direct'] = {'requests': len(direct), 'errors': errors,
                            'p50_seconds': statistics.median(latency),
                            'p95_seconds': percentile(latency, 0.95),
                            'error_rate': errors / len(direct)}
        if report['direct']['p95_seconds'] > args.max_direct_p95:
            report['failures'].append(f"direct p95 {report['direct']['p95_seconds']:.2f}s exceeds {args.max_direct_p95}s")
        if report['direct']['error_rate'] > args.max_error_rate:
            report['failures'].append(f"direct error rate {report['direct']['error_rate']:.2%} exceeds {args.max_error_rate:.2%}")

        # 2) Concurrent session isolation: interleave two sessions on the pronoun path.
        sessions = [str(uuid.uuid4()), str(uuid.uuid4())]
        first = ['What is metformin used for?', 'What is warfarin used for?']
        def converse(i):
            return [chat(client, url, first[i], sessions[i]), chat(client, url, FOLLOWUP, sessions[i])]
        with ThreadPoolExecutor(max_workers=2) as pool:
            conversations = list(pool.map(converse, range(2)))
        report['session_isolation'] = {
            'sessions': sessions,
            'pass': all(drugs[0]['status'] == 200 and drugs[1]['status'] == 200
                        and expected in drugs[1]['answer'].lower()
                        for drugs, expected in zip(conversations, ('metformin', 'warfarin')))}
        if not report['session_isolation']['pass']:
            report['failures'].append('session isolation failed: follow-up used the wrong drug')

        # 3) Agent-route latency and queue behavior (only when requested).
        if args.agent_question:
            with ThreadPoolExecutor(max_workers=args.concurrency + 2) as pool:
                agent_rows = list(pool.map(lambda _: chat(client, url, args.agent_question),
                                           range(args.concurrency + 2)))
            agent_latency = sorted(r['seconds'] for r in agent_rows if r['status'] == 200)
            queued_out = sum(r['status'] == 503 for r in agent_rows)
            report['agent'] = {'requests': len(agent_rows), 'ok': len(agent_latency),
                               'queue_rejections_503': queued_out,
                               'p50_seconds': statistics.median(agent_latency) if agent_latency else None,
                               'p95_seconds': percentile(agent_latency, 0.95) if agent_latency else None}
            if agent_latency and report['agent']['p95_seconds'] > args.max_agent_p95:
                report['failures'].append(f"agent p95 {report['agent']['p95_seconds']:.1f}s exceeds {args.max_agent_p95}s")
            if not queued_out:
                report['failures'].append('queue saturation was not exercised: expected some 503s')

        after = client.get(url + '/api/metrics').json()
        report['memory'] = {'before_mb': before['memory_rss_mb'], 'after_mb': after['memory_rss_mb'],
                            'growth_mb': round(after['memory_rss_mb'] - before['memory_rss_mb'], 1)}
        report['server_requests'] = after['requests']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
