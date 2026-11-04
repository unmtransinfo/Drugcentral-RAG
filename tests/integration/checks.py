"""In-network checks for the local-mode smoke test; run by run.sh inside the api image. Never prints the token.

    python tests/integration/checks.py isolation   # no AWS configuration, no outside network
    python tests/integration/checks.py embedding   # Ollama embedding through the application's factory
    python tests/integration/checks.py api         # health, readiness, /docs, auth, chat, metrics
"""
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip(), flush=True)


def isolation():
    check('no AWS_* variables', not any(k.startswith('AWS_') for k in os.environ))
    check('no ~/.aws directory', not os.path.exists(os.path.expanduser('~/.aws')))
    check('no AWS SDK installed', importlib.util.find_spec('boto3') is None)
    for name, host, port in [('EC2 instance metadata', '169.254.169.254', 80), ('internet', '1.1.1.1', 443)]:
        try:
            socket.create_connection((host, port), timeout=3).close()
            check(f'{name} unreachable', False, 'reachable')
        except OSError:
            check(f'{name} unreachable', True)


def embedding():
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    vector = create_embeddings().embed_query('metformin indication')
    check('embedding', len(vector) == 768, f'{embedding_model_id()} dimensions={len(vector)}')


def api():
    import httpx
    token = os.environ['API_AUTH_TOKEN']
    auth = {'Authorization': f'Bearer {token}'}
    with httpx.Client(base_url=os.environ.get('ITEST_API', 'http://api:8000'), timeout=1200, trust_env=False) as c:
        for _ in range(90):
            try:
                if c.get('/api/ready').status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(2)
        r = c.get('/api/health')
        check('GET /api/health', r.status_code == 200, f'{r.status_code}')
        r = c.get('/api/ready')
        body = r.json() if r.status_code in (200, 503) else {}
        check('GET /api/ready', r.status_code == 200 and body.get('provider') == 'ollama' and body.get('drugs') == 3,
              f'{r.status_code} {json.dumps(body)}')
        r = c.get('/docs')
        check('GET /docs (Swagger UI)', r.status_code == 200 and 'swagger-ui' in r.text, f'{r.status_code}')
        r = c.get('/openapi.json')
        check('GET /openapi.json declares bearer auth on /api/chat',
              r.status_code == 200 and 'security' in r.json()['paths']['/api/chat']['post'], f'{r.status_code}')
        question = {'question': 'What is metformin indicated for?'}
        r = c.post('/api/chat', json=question)
        check('POST /api/chat without token -> 401', r.status_code == 401, f'{r.status_code}')
        r = c.post('/api/chat', json=question, headers={'Authorization': 'Bearer wrong-token'})
        check('POST /api/chat with wrong token -> 401', r.status_code == 401, f'{r.status_code}')
        started = time.monotonic()
        r = c.post('/api/chat', json=question, headers=auth)
        answer = r.json().get('answer', '') if r.status_code == 200 else r.text
        check('POST /api/chat with token -> grounded answer with Sources',
              r.status_code == 200 and 'diabetes' in answer.lower() and 'Sources:' in answer,
              f'{r.status_code} {time.monotonic() - started:.0f}s')
        print('      answer:', answer.replace('\n', ' ')[:400], flush=True)
        r = c.get('/api/metrics', headers=auth)
        check('GET /api/metrics with token', r.status_code == 200, f'{r.status_code}')


if __name__ == '__main__':
    {'isolation': isolation, 'embedding': embedding, 'api': api}[sys.argv[1]]()
    sys.exit(0 if results and all(results) else 1)
