"""Ask a running DrugCentral RAG API from the command line (developer tool; standard library only).

    export API_AUTH_TOKEN=...                       # read from the environment, never printed or passed as an argument
    python -m cli.ask "What are the common side effects of metformin?"
    python -m cli.ask                               # interactive; follow-up questions share one session
    python -m cli.ask --check                       # /api/health and /api/ready only
    echo "Which drugs target EGFR?" | python -m cli.ask

Run from the repository root. The API URL comes from --url, else DRUGCENTRAL_API_URL, else http://127.0.0.1:8000
(the API binds to localhost; use an SSH tunnel to reach a remote host). Exit status: 0 success, 1 request failed,
2 usage error.
"""
import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid

from evaluation.quick_check import classify

DEFAULT_URL = 'http://127.0.0.1:8000'
SOURCE_LINK = re.compile(r'(?:\[(\d+)\]\s*)?\[([^\]]+)\]\((\S+?)\)')


class ApiError(Exception):
    """A request failed; the message is safe to show and never contains the token."""


def split_answer(answer):
    """Answer body and a list of (number, label, url) sources parsed from the trailing 'Sources:' line."""
    parts = re.split(r'\n\s*Sources:\s*', answer or '', maxsplit=1)
    body = parts[0].strip()
    sources = [(n or str(i), label, url) for i, (n, label, url) in enumerate(SOURCE_LINK.findall(parts[1]), 1)] \
        if len(parts) == 2 else []
    if len(parts) == 2 and not sources:
        body = (answer or '').strip()      # unrecognised footer: show it unchanged rather than drop it
    return body, sources


def format_answer(answer):
    body, sources = split_answer(answer)
    status = classify(answer)
    lines = [body]
    if status != 'answered':
        lines.insert(0, f'[{status}]')
    if sources:
        lines.append('')
        lines.append('Sources:')
        lines.extend(f'  [{n}] {label}\n      {url}' for n, label, url in sources)
    return '\n'.join(lines)


def _detail(error):
    try:
        detail = json.loads(error.read().decode('utf-8', 'replace')).get('detail')
    except (ValueError, AttributeError, OSError):
        detail = None
    if isinstance(detail, list):        # FastAPI validation errors
        detail = '; '.join(str(item.get('msg', item)) for item in detail)
    return detail or error.reason


def request(url, path, *, token=None, payload=None, timeout=180.0, retries=2):
    """GET (payload None) or POST JSON; retries 429/503 with Retry-After; returns parsed JSON."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {'Accept': 'application/json'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    if token:
        headers['Authorization'] = 'Bearer ' + token
    target = url.rstrip('/') + path
    for attempt in range(retries + 1):
        req = urllib.request.Request(target, data=data, headers=headers, method='POST' if data is not None else 'GET')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as error:
            detail = _detail(error)
            if error.code in (429, 503) and attempt < retries:
                time.sleep(min(float(error.headers.get('Retry-After') or 5 * (attempt + 1)), 30))
                continue
            if error.code == 401:
                raise ApiError('401 Unauthorized: set API_AUTH_TOKEN (or the variable named by --token-env) '
                               'to the token configured on the server') from None
            if error.code == 422:
                raise ApiError(f'422 Invalid request: {detail}') from None
            raise ApiError(f'{error.code} {detail}') from None
        except (socket.timeout, TimeoutError):
            raise ApiError(f'No response within {timeout:g} s from {target} (raise --timeout for long answers)') from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise ApiError(f'No response within {timeout:g} s from {target} (raise --timeout for long answers)') from None
            raise ApiError(f'Cannot reach the API at {url} ({error.reason}); is it running, '
                           'and is the SSH tunnel open?') from None
        except ValueError:
            raise ApiError(f'The response from {target} was not JSON; is --url pointing at this API?') from None
    raise ApiError('The assistant stayed busy; try again shortly')


def ask(url, question, *, token, session_id=None, timeout=180.0, retries=2):
    payload = {'question': question}
    if session_id:
        payload['session_id'] = session_id
    answer = request(url, '/api/chat', token=token, payload=payload, timeout=timeout, retries=retries).get('answer')
    if not isinstance(answer, str):
        raise ApiError('The response did not contain an answer')
    return answer


def check(url, timeout):
    health = request(url, '/api/health', timeout=timeout, retries=0)
    print(f'health: {health.get("status", health)}')
    ready = request(url, '/api/ready', timeout=timeout, retries=0)
    versions = ', '.join(ready.get('dataset_version', [])) or '-'
    print(f'ready:  {ready.get("status")}  provider={ready.get("provider", "-")}  drugs={ready.get("drugs", "-")}  '
          f'dataset={versions}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ask a running DrugCentral RAG API.',
                                     epilog='The token is read from the environment only; it is never printed.')
    parser.add_argument('question', nargs='*', help='question text; omit for interactive mode or piped input')
    parser.add_argument('--url', default=os.getenv('DRUGCENTRAL_API_URL', DEFAULT_URL),
                        help=f'API base URL (default: $DRUGCENTRAL_API_URL or {DEFAULT_URL})')
    parser.add_argument('--token-env', default='API_AUTH_TOKEN', metavar='NAME',
                        help='environment variable holding the bearer token (default: API_AUTH_TOKEN)')
    parser.add_argument('--timeout', type=float, default=180.0, help='seconds to wait per request (default: 180)')
    parser.add_argument('--retries', type=int, default=2, help='retries when the API is busy (429/503; default: 2)')
    parser.add_argument('--no-session', action='store_true', help='do not share a session between interactive questions')
    parser.add_argument('--raw', action='store_true', help='print the answer text exactly as returned')
    parser.add_argument('--check', action='store_true', help='only call /api/health and /api/ready')
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries < 0:
        parser.error('--timeout must be positive and --retries non-negative')

    try:
        if args.check:
            check(args.url, args.timeout)
            return 0
        token = os.environ.get(args.token_env, '').strip() or None
        if token is None:
            print(f'note: {args.token_env} is not set; sending the request without a token', file=sys.stderr)

        def answer(question, session_id=None):
            text = ask(args.url, question, token=token, session_id=session_id, timeout=args.timeout, retries=args.retries)
            print(text if args.raw else format_answer(text))

        if args.question:
            answer(' '.join(args.question))
            return 0
        session_id = None if args.no_session else str(uuid.uuid4())
        interactive = sys.stdin.isatty()
        if interactive:
            print(f'Asking {args.url}. Empty line or Ctrl-D to quit.', file=sys.stderr)
        failed = False
        while True:
            try:
                line = input('question> ' if interactive else '')
            except EOFError:
                break
            if not line.strip():
                if interactive:
                    break
                continue
            try:
                answer(line.strip(), session_id)
            except ApiError as error:
                print(f'error: {error}', file=sys.stderr)
                failed = True
            if interactive:
                print()
        return 1 if failed else 0
    except ApiError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
