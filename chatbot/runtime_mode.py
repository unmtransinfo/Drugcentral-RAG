"""Runtime mode selection: RAG_PIPELINE=local (Ollama) or verified-bedrock (Amazon Bedrock).

The mode decides the default chat and embedding backends. Explicit CHAT_BACKEND/EMBED_BACKEND values
must agree with it, so a configuration can never mix Ollama and Bedrock silently.
"""
import os

PIPELINES = {
    'local': {'chat': 'ollama', 'embed': 'ollama'},
    'verified-bedrock': {'chat': 'bedrock', 'embed': 'bedrock'},
}
BACKENDS = {'ollama', 'bedrock'}
EXAMPLES = 'see environments/local/ollama.env.example or environments/local/bedrock.env.example'


def require_package(module, mode, lock):
    """Import a mode-specific dependency, or explain which dependency group to install."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        raise RuntimeError(f'{mode} needs the {module!r} package: pip install -r {lock}') from None


def configured_pipeline():
    """The selected pipeline, or None when RAG_PIPELINE is unset (ingestion tools may run without it)."""
    value = os.getenv('RAG_PIPELINE', '').strip().lower()
    if not value:
        return None
    if value not in PIPELINES:
        raise ValueError(f"RAG_PIPELINE={value!r} is not supported; use 'local' (Ollama) or "
                         f"'verified-bedrock' (Amazon Bedrock); {EXAMPLES}")
    return value


def pipeline():
    """The pipeline the API serves; required."""
    value = configured_pipeline()
    if value is None:
        raise ValueError(f"Set RAG_PIPELINE to 'local' (Ollama) or 'verified-bedrock' (Amazon Bedrock); {EXAMPLES}")
    return value


def backend(kind, variable):
    """Resolve the chat or embedding backend from its variable and the pipeline; reject conflicts."""
    selected = configured_pipeline()
    default = PIPELINES[selected][kind] if selected else None
    value = os.getenv(variable, '').strip().lower() or default
    if not value:
        raise ValueError(f'Set {variable} (ollama or bedrock) or RAG_PIPELINE; {EXAMPLES}')
    if value not in BACKENDS:
        raise ValueError(f'{variable}={value!r} is not supported; use ollama or bedrock '
                         '(MedCPT, Titan and other backends are not part of this repository)')
    if default and value != default:
        raise ValueError(f'{variable}={value!r} conflicts with RAG_PIPELINE={selected!r}, which requires {default}')
    return value
