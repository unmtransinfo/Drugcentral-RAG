"""cli.ask against a local stub server: auth from the environment, readable sources, clear errors."""

import contextlib
import io
import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from cli import ask

TOKEN = 'stub-token-value'
ANSWER = ('Metformin commonly causes diarrhea [1] and nausea [2].\n\n'
          'Sources: [1] [metformin label](https://example.org/a) · [2] [DrugCentral drug card](https://example.org/b)')


class Stub(BaseHTTPRequestHandler):
    busy_once = False
    delay = 0
    seen = []

    def log_message(self, *args):
        pass

    def _send(self, code, body, headers=()):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        for key, value in headers:
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/api/health':
            self._send(200, {'status': 'ok'})
        elif self.path == '/api/ready':
            self._send(200, {'status': 'ready', 'provider': 'bedrock', 'drugs': 5116, 'dataset_version': ['v1']})
        else:
            self._send(404, {'detail': 'Not Found'})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Stub.seen.append((self.headers.get('Authorization'), body))
        time.sleep(Stub.delay)
        if self.headers.get('Authorization') != 'Bearer ' + TOKEN:
            self._send(401, {'detail': 'Unauthorized'}, [('WWW-Authenticate', 'Bearer')])
        elif Stub.busy_once:
            Stub.busy_once = False
            self._send(503, {'detail': 'The assistant is busy.'}, [('Retry-After', '0')])
        else:
            self._send(200, {'answer': ANSWER})


class CliAskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        Stub.busy_once, Stub.delay, Stub.seen = False, 0, []

    def run_cli(self, *argv, env=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env or {}, clear=False), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            os.environ.pop('API_AUTH_TOKEN', None) if env is None else None
            code = ask.main(['--url', self.url, *argv])
        return code, out.getvalue(), err.getvalue()

    def test_answer_with_sources_and_token_from_environment(self):
        code, out, err = self.run_cli('What', 'about', 'metformin?', env={'API_AUTH_TOKEN': TOKEN})
        self.assertEqual(code, 0)
        self.assertIn('Metformin commonly causes diarrhea [1]', out)
        self.assertIn('  [1] metformin label\n      https://example.org/a', out)
        self.assertIn('  [2] DrugCentral drug card', out)
        self.assertEqual(Stub.seen[0], ('Bearer ' + TOKEN, {'question': 'What about metformin?'}))
        self.assertNotIn(TOKEN, out + err)

    def test_token_variable_name_is_configurable(self):
        code, _, _ = self.run_cli('--token-env', 'MY_TOKEN', 'q', env={'MY_TOKEN': TOKEN})
        self.assertEqual(code, 0)

    def test_missing_token_gives_clear_401_message(self):
        code, out, err = self.run_cli('q')
        self.assertEqual(code, 1)
        self.assertIn('401 Unauthorized: set API_AUTH_TOKEN', err)
        self.assertIsNone(Stub.seen[0][0])

    def test_busy_api_is_retried(self):
        Stub.busy_once = True
        code, out, _ = self.run_cli('q', env={'API_AUTH_TOKEN': TOKEN})
        self.assertEqual(code, 0)
        self.assertEqual(len(Stub.seen), 2)

    def test_timeout_is_reported(self):
        Stub.delay = 1.5
        code, _, err = self.run_cli('--timeout', '0.3', 'q', env={'API_AUTH_TOKEN': TOKEN})
        self.assertEqual(code, 1)
        self.assertIn('No response within 0.3 s', err)

    def test_unreachable_api_is_reported(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ask.main(['--url', 'http://127.0.0.1:9', '--timeout', '2', 'q'])
        self.assertEqual(code, 1)
        self.assertIn('Cannot reach the API at http://127.0.0.1:9', err.getvalue())

    def test_piped_questions_share_one_session(self):
        with patch('sys.stdin', io.StringIO('first question\nsecond question\n')):
            code, out, _ = self.run_cli(env={'API_AUTH_TOKEN': TOKEN})
        self.assertEqual(code, 0)
        sessions = {body.get('session_id') for _, body in Stub.seen}
        self.assertEqual(len(Stub.seen), 2)
        self.assertEqual(len(sessions), 1)
        self.assertIsNotNone(sessions.pop())

    def test_check_reports_health_and_readiness(self):
        code, out, _ = self.run_cli('--check')
        self.assertEqual(code, 0)
        self.assertIn('health: ok', out)
        self.assertIn('ready:  ready  provider=bedrock  drugs=5116  dataset=v1', out)

    def test_abstention_is_labelled_and_unknown_footer_kept(self):
        self.assertTrue(ask.format_answer('The requested information could not be verified.').startswith('[abstained]'))
        self.assertIn('Sources: see label', ask.format_answer('Body.\n\nSources: see label'))


if __name__ == '__main__':
    unittest.main()
