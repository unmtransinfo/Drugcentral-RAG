"""Chat-provider configuration: Ollama (RAG_PIPELINE=local) or Amazon Bedrock (verified-bedrock).

AWS access always comes from the standard AWS credential chain of the environment running the code.
"""
import os

from chatbot.runtime_mode import backend


def chat_backend():
    return backend('chat', 'CHAT_BACKEND')


def chat_model_id():
    if chat_backend() == "ollama":
        return os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b")
    model = os.getenv("BEDROCK_GENERATION_MODEL", "").strip()
    if not model:
        raise ValueError("BEDROCK_GENERATION_MODEL is required")
    return model


def aws_region():
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS_REGION is required for Bedrock")
    return region


def create_chat_model(timeout_seconds=None, max_tokens=None):
    """Ollama chat model for the local pipeline (verified-bedrock calls Bedrock Converse directly)."""
    if chat_backend() != "ollama":
        raise ValueError("The local answer pipeline uses Ollama; Bedrock generation runs through RAG_PIPELINE=verified-bedrock")
    model = chat_model_id()
    from chatbot.runtime_mode import require_package
    ChatOllama = require_package('langchain_ollama', 'Local mode (Ollama)', 'environments/local/requirements-local.lock').ChatOllama
    return ChatOllama(model=model, temperature=0, keep_alive="30m",
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        num_predict=max_tokens or int(os.getenv("OLLAMA_NUM_PREDICT", "1200")),
        client_kwargs={"timeout": timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))},
        repeat_penalty=1.1,
        **({"reasoning": False} if model.startswith(("qwen3", "deepseek-r1")) else {}))
