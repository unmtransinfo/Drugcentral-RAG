"""Mode selection, collection safety and model request formats for both modes.

No models, AWS access or databases are used: Bedrock runtimes and database cursors are stubs. Tests that need a
package from only one dependency group (langchain-ollama for local mode, boto3 for Bedrock mode) skip otherwise.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from chatbot.embedding_config import embedding_backend, embedding_model_id
from chatbot.model_config import chat_backend, chat_model_id
from chatbot.runtime_mode import pipeline
from chatbot.storage import _check_embedding_identity, _refuse_identity_change, backend_name

HAS_OLLAMA = importlib.util.find_spec('langchain_ollama') is not None
HAS_BOTO3 = importlib.util.find_spec('boto3') is not None
SELECTORS = ('RAG_PIPELINE', 'EMBED_BACKEND', 'CHAT_BACKEND', 'VECTOR_BACKEND')
BEDROCK = {'RAG_PIPELINE': 'verified-bedrock', 'AWS_REGION': 'us-east-2', 'BEDROCK_EMBED_MODEL': 'us.cohere.embed-v4:0',
           'BEDROCK_EMBED_DIMENSIONS': '1024', 'BEDROCK_RERANK_MODEL': 'cohere.rerank-v3-5:0',
           'BEDROCK_GENERATION_MODEL': 'qwen.qwen3-next-80b-a3b', 'BEDROCK_VALIDATION_MODEL': 'qwen.qwen3-next-80b-a3b'}
COHERE_IDENTITY = 'bedrock:us.cohere.embed-v4:0:dim=1024:document=search_document:query=search_query:float:truncate=NONE'


class Env(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {})
        self.env.start()
        for name in SELECTORS + tuple(BEDROCK):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()


class ModeSelectionTests(Env):
    def test_pipeline_is_required(self):
        with self.assertRaisesRegex(ValueError, "Set RAG_PIPELINE to 'local'"):
            pipeline()

    def test_local_mode_defaults_to_ollama(self):
        os.environ['RAG_PIPELINE'] = 'local'
        self.assertEqual((chat_backend(), embedding_backend(), backend_name()), ('ollama', 'ollama', 'pgvector'))
        self.assertEqual((chat_model_id(), embedding_model_id()), ('qwen3.5:9b', 'nomic-embed-text'))

    def test_bedrock_mode_defaults_to_bedrock(self):
        os.environ.update(BEDROCK)
        self.assertEqual((chat_backend(), embedding_backend()), ('bedrock', 'bedrock'))
        self.assertEqual((chat_model_id(), embedding_model_id()), ('qwen.qwen3-next-80b-a3b', COHERE_IDENTITY))

    def test_mixed_backends_are_rejected(self):
        for mode, name, value in [('local', 'CHAT_BACKEND', 'bedrock'), ('local', 'EMBED_BACKEND', 'bedrock'),
                                  ('verified-bedrock', 'CHAT_BACKEND', 'ollama'), ('verified-bedrock', 'EMBED_BACKEND', 'ollama')]:
            with self.subTest(mode=mode, name=name), patch.dict(os.environ, {'RAG_PIPELINE': mode, name: value}):
                with self.assertRaisesRegex(ValueError, f"{name}='{value}' conflicts with RAG_PIPELINE='{mode}'"):
                    (chat_backend if name == 'CHAT_BACKEND' else embedding_backend)()

    def test_unsupported_values_are_rejected(self):
        cases = [({'RAG_PIPELINE': 'legacy'}, pipeline, "RAG_PIPELINE='legacy' is not supported"),
                 ({'EMBED_BACKEND': 'other'}, embedding_backend, "EMBED_BACKEND='other' is not supported"),
                 ({'CHAT_BACKEND': 'other'}, chat_backend, "CHAT_BACKEND='other' is not supported"),
                 ({'VECTOR_BACKEND': 'other'}, backend_name, "VECTOR_BACKEND='other' is not supported")]
        for env, check, message in cases:
            with self.subTest(env=env), patch.dict(os.environ, env):
                with self.assertRaisesRegex(ValueError, message):
                    check()

    def test_bedrock_model_ids_are_required(self):
        os.environ['RAG_PIPELINE'] = 'verified-bedrock'
        with self.assertRaisesRegex(ValueError, 'BEDROCK_EMBED_MODEL is required'):
            embedding_model_id()
        with self.assertRaisesRegex(ValueError, 'BEDROCK_GENERATION_MODEL is required'):
            chat_model_id()


class FakeCursor:
    """Answers the catalogue and stamp queries used by the collection guards."""

    def __init__(self, stamp=None, chunk=None, has_metadata=True):
        self.stamp, self.chunk, self.has_metadata, self.last = stamp, chunk, has_metadata, ''

    def execute(self, sql, params=None):
        self.last = sql

    def fetchone(self):
        if 'to_regclass' in self.last:
            return ('public.drugcentral_collections' if self.has_metadata else None,)
        if 'FROM public.drugcentral_chunks' in self.last:
            return self.chunk
        if 'FROM public.drugcentral_collections' in self.last:
            return self.stamp
        return None


class CollectionSafetyTests(Env):
    def test_local_mode_refuses_a_bedrock_collection(self):
        os.environ.update({'RAG_PIPELINE': 'local', 'VECTOR_COLLECTION': 'shared'})
        cursor = FakeCursor(stamp=('bedrock', COHERE_IDENTITY, '', 1024, None), chunk=(0, 'text', '[0.1,0.2]'))
        with self.assertRaisesRegex(ValueError, "backend stamped 'bedrock' but configured 'ollama'"):
            _check_embedding_identity(cursor, 'nomic-embed-text', None, read_only=True)

    def test_bedrock_mode_refuses_an_ollama_collection(self):
        os.environ.update(BEDROCK, VECTOR_COLLECTION='shared')
        cursor = FakeCursor(stamp=('ollama', 'nomic-embed-text', '', 768, None), chunk=(0, 'text', '[0.1,0.2]'))
        with self.assertRaisesRegex(ValueError, "backend stamped 'ollama' but configured 'bedrock'"):
            _check_embedding_identity(cursor, COHERE_IDENTITY, None, read_only=True)

    def test_ingestion_never_overwrites_another_embedders_collection(self):
        os.environ.update({'RAG_PIPELINE': 'local', 'VECTOR_COLLECTION': 'drugcentral_dev_cohere_v4_1024'})
        with self.assertRaisesRegex(ValueError, 'holds bedrock vectors .* Choose a different VECTOR_COLLECTION'):
            _refuse_identity_change(FakeCursor(stamp=('bedrock', COHERE_IDENTITY)), 'nomic-embed-text')

    def test_ingestion_may_rebuild_its_own_or_a_new_collection(self):
        os.environ.update({'RAG_PIPELINE': 'local', 'VECTOR_COLLECTION': 'drugcentral_local_nomic_embed_text'})
        _refuse_identity_change(FakeCursor(stamp=('ollama', 'nomic-embed-text')), 'nomic-embed-text')
        _refuse_identity_change(FakeCursor(stamp=None), 'nomic-embed-text')
        _refuse_identity_change(FakeCursor(has_metadata=False), 'nomic-embed-text')


@unittest.skipUnless(HAS_OLLAMA, 'local dependency group (langchain-ollama) not installed')
class LocalModeTests(unittest.TestCase):
    def test_local_mode_builds_ollama_clients_without_importing_aws(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('AWS_', 'BEDROCK_')) and k not in SELECTORS}
        env.update(RAG_PIPELINE='local', AWS_EC2_METADATA_DISABLED='true')
        code = ('import sys\n'
                'from chatbot.embedding_config import create_embeddings\n'
                'from chatbot.model_config import create_chat_model\n'
                'import api, chatbot.vector_answers\n'
                'e = create_embeddings(timeout_seconds=10); m = create_chat_model()\n'
                'print(type(e).__name__, type(m).__name__, any(n.split(".")[0] in ("boto3", "botocore") for n in sys.modules))')
        out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ['OllamaEmbeddings', 'ChatOllama', 'False'])


class StubBody(io.BytesIO):
    pass


class StubRuntime:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {'body': StubBody(json.dumps(self.payload).encode())}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


@unittest.skipUnless(HAS_BOTO3, 'Bedrock dependency group (boto3) not installed')
class BedrockRequestTests(Env):
    def setUp(self):
        super().setUp()
        os.environ.update(BEDROCK)

    def test_cohere_embedding_request_format(self):
        from chatbot.production_models import CohereEmbeddings
        runtime = StubRuntime({'embeddings': {'float': [[0.5] * 1024]}})
        vector = CohereEmbeddings(runtime=runtime).embed_query('metformin')
        body = json.loads(runtime.calls[0]['body'])
        self.assertEqual(runtime.calls[0]['modelId'], 'us.cohere.embed-v4:0')
        self.assertEqual((body['input_type'], body['output_dimension'], body['truncate']), ('search_query', 1024, 'NONE'))
        self.assertEqual(len(vector), 1024)

    def test_cohere_rerank_request_format(self):
        from chatbot.production_models import rerank
        runtime = StubRuntime({'results': [{'index': 1, 'relevance_score': 0.9}, {'index': 0, 'relevance_score': 0.2}]})
        records = [{'text': 'a', 'metadata': {'drug_name': 'x', 'section': 's'}},
                   {'text': 'b', 'metadata': {'drug_name': 'y', 'section': 's'}}]
        ranked = rerank('question', records, runtime=runtime)
        body = json.loads(runtime.calls[0]['body'])
        self.assertEqual((runtime.calls[0]['modelId'], body['api_version'], body['top_n']), ('cohere.rerank-v3-5:0', 2, 2))
        self.assertEqual([r['text'] for r in ranked], ['b', 'a'])

    def test_generation_and_validation_use_converse(self):
        from chatbot.production_models import converse
        runtime = StubRuntime({'stopReason': 'end_turn', 'usage': {},
                               'output': {'message': {'content': [{'text': '{"ok": true}'}]}}})
        metrics = {}
        self.assertEqual(converse('system', {'q': 1}, 'generation', metrics, runtime=runtime), {'ok': True})
        self.assertEqual(converse('system', {'q': 1}, 'validation', metrics, runtime=runtime), {'ok': True})
        self.assertEqual([c['modelId'] for c in runtime.calls], ['qwen.qwen3-next-80b-a3b'] * 2)

    def test_bedrock_client_uses_the_default_credential_chain(self):
        from chatbot import production_models
        production_models.client.cache_clear()
        with patch('boto3.Session') as session:
            production_models.client('us-east-2')
        session.assert_called_once_with()                       # no explicit keys, profile or account
        args, kwargs = session.return_value.client.call_args
        self.assertEqual((args[0], kwargs['region_name']), ('bedrock-runtime', 'us-east-2'))
        production_models.client.cache_clear()


if __name__ == '__main__':
    unittest.main()
