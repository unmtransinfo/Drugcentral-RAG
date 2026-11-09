"""Alert gate for scheduled jobs (e.g. cron) or external monitoring.

Checks readiness, 5xx error rate, and chat p95 against limits; exits 1 with a
human-readable message when any limit is breached. Pair with API_AUTH_TOKEN
when auth is enabled.

Example cron (every 5 minutes, mail/log on failure):
*/5 * * * * python environments/prod/check_alerts.py --url http://127.0.0.1:8000
"""
import argparse
import json
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--token', default=None, help='Bearer token; defaults to API_AUTH_TOKEN env')
    parser.add_argument('--max-error-rate', type=float, default=0.05)
    parser.add_argument('--max-p95', type=float, default=30.0)
    args = parser.parse_args()
    import os
    token = args.token or os.getenv('API_AUTH_TOKEN', '')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    url = args.url.rstrip('/')
    failures = []
    with httpx.Client(timeout=10, trust_env=False, headers=headers) as client:
        try:
            ready = client.get(url + '/api/ready')
            if ready.status_code != 200:
                failures.append(f'readiness returned {ready.status_code}')
        except httpx.HTTPError as exc:
            failures.append(f'readiness unreachable: {type(exc).__name__}')
            print(json.dumps({'ok': False, 'failures': failures}))
            raise SystemExit(1)
        metrics = client.get(url + '/api/metrics').json()
    counts = metrics.get('requests', {})
    total = sum(counts.get(f'chat:{code}', 0) for code in (200, 422, 500, 503))
    errors = sum(count for name, count in counts.items()
                 if name.startswith('chat:') and int(name.rsplit(':', 1)[1]) >= 500)
    if total and errors / total > args.max_error_rate:
        failures.append(f'5xx error rate {errors / total:.2%} exceeds {args.max_error_rate:.2%}')
    p95 = metrics.get('chat_seconds', {}).get('p95', 0.0)
    if metrics.get('chat_seconds', {}).get('count', 0) >= 20 and p95 > args.max_p95:
        failures.append(f'chat p95 {p95:.2f}s exceeds {args.max_p95}s over the recorded window')
    print(json.dumps({'ok': not failures, 'failures': failures,
                      'requests': counts, 'chat_seconds': metrics.get('chat_seconds'),
                      'memory_rss_mb': metrics.get('memory_rss_mb')}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
