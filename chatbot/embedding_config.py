"""Shared embedding identity and construction for ingestion and retrieval.

Ollama (RAG_PIPELINE=local) and Cohere Embed on Bedrock (verified-bedrock) produce different, incompatible
vectors. Each collection is stamped with its backend and model identity and is only served by that identity.
"""
import os

from chatbot.runtime_mode import backend


def embedding_backend():
    return backend('embed', 'EMBED_BACKEND')


def embedding_model_id():
    if embedding_backend() == 'bedrock':
        if not os.getenv('BEDROCK_EMBED_MODEL', '').strip():
            raise ValueError('BEDROCK_EMBED_MODEL is required (for example us.cohere.embed-v4:0)')
        from chatbot.production_models import embed_identity
        return embed_identity()
    return os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


def create_embeddings(timeout_seconds=None):
    if embedding_backend() == 'bedrock':
        embedding_model_id()
        from chatbot.production_models import CohereEmbeddings
        return CohereEmbeddings()
    from chatbot.runtime_mode import require_package
    OllamaEmbeddings = require_package('langchain_ollama', 'Local mode (Ollama)',
                                       'environments/local/requirements-local.lock').OllamaEmbeddings
    return OllamaEmbeddings(model=embedding_model_id(), **({'client_kwargs': {'timeout': timeout_seconds}} if timeout_seconds is not None else {}))
