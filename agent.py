"""Application startup and the two answer pipelines: local (Ollama) and verified-bedrock (Amazon Bedrock)."""
import os
from dotenv import load_dotenv
from chatbot.storage import backend_name, load_documents, search_documents
from chatbot.embedding_config import create_embeddings, embedding_backend, embedding_model_id
from chatbot.retrieval import EvidenceIndex

load_dotenv()
if os.getenv('LABEL_REQUIRED') == '1':
    from chatbot.label_retrieval import database_path
    if not database_path().is_file():
        raise RuntimeError('Configured label database is missing; build it or mount LABEL_DATABASE')

from chatbot.runtime_mode import pipeline
from chatbot.model_config import chat_backend
PIPELINE = pipeline()
backend_name()  # rejects any VECTOR_BACKEND other than pgvector
chat_backend(), embedding_backend()  # reject backends that conflict with RAG_PIPELINE
embeddings_model = create_embeddings(timeout_seconds=10)
EMBEDDING_MODEL_ID = embedding_model_id()
if PIPELINE == 'verified-bedrock':
    if not os.getenv('BEDROCK_EMBED_MODEL'):
        raise ValueError('verified-bedrock requires pgvector, Bedrock generation and Cohere embeddings')
    from chatbot.hybrid_search import limits
    from chatbot.production_models import validate_settings
    validate_settings()
    limits()
    from chatbot.storage import _connect, _collection
    from contextlib import closing
    with closing(_connect()) as conn, conn, conn.cursor() as cur:
        cur.execute('SELECT model FROM public.drugcentral_collections WHERE collection=%s', (_collection(),))
        stamp = cur.fetchone()
        if not stamp or stamp[0] != EMBEDDING_MODEL_ID:
            raise ValueError(f'Collection {_collection()!r} has no completed embedding stamp for this model: it does not exist, is still being embedded, or VECTOR_COLLECTION is wrong. Complete and validate the Cohere migration before activating it')
        cur.execute('SELECT source,status FROM public.drugcentral_migrations WHERE collection=%s', (_collection(),))
        migration = cur.fetchone()
        if migration != (os.getenv('STRUCTURED_COLLECTION'), 'complete'):
            raise ValueError(f'Vector collection {_collection()!r} was built from {migration[0] if migration else None!r}, not STRUCTURED_COLLECTION={os.getenv("STRUCTURED_COLLECTION")!r}, or its migration is not complete')
print(f'Loading DrugCentral records from {backend_name()}...')
evidence_index = EvidenceIndex(load_documents(embeddings_model, EMBEDDING_MODEL_ID))
# Compatibility for API/evaluation callers; no unused tool executor is built.
agent_executor = None


def invoke_english_only(executor, query: str, max_retries: int = 1) -> str:
    """Answer one question with the configured pipeline."""
    from chatbot.intent import (classify as classify_intent,
                                vector_sections as intent_vector_sections,
                                label_sections as intent_label_sections,
                                apply_relevance_floor)
    from chatbot.label_retrieval import search_labels
    if PIPELINE == 'verified-bedrock':
        from chatbot.production_pipeline import answer
        from chatbot.question_retrieval import requested_sections
        label_fields = {'adverse_events', 'adverse_reactions', 'warnings', 'precautions', 'dosage'}
        classified = classify_intent(query)
        wants_labels = bool(label_fields & set(requested_sections(query))) or bool(
            classified and intent_label_sections(classified))
        return answer(evidence_index, query, embeddings_model, EMBEDDING_MODEL_ID,
                      label_lookup=(lambda q, d: search_labels(evidence_index, q, d, exact_text=True)) if wants_labels else None)
    # Local pipeline: Ollama generation over the local pgvector collection.
    from chatbot.vector_answers import vector_answer, OUTPUT_SCHEMA, ENTAILMENT_SYSTEM, ENTAILMENT_SCHEMA
    from chatbot.model_config import create_chat_model
    intent_name = classify_intent(query)
    # Product labels are consulted only for intents whose fields are label-sourced;
    # a formulation clarification must never hijack a snapshot-answerable question.
    label_lookup = (lambda q, d: search_labels(evidence_index, q, d)
                    ) if (intent_name and intent_label_sections(intent_name)) else None
    similarity = os.getenv('RETRIEVAL_MIN_SIMILARITY', '').strip()
    similarity_kwargs = {'min_similarity': float(similarity)} if similarity else {}
    def retrieve_vectors(question, drug):
        from chatbot.question_retrieval import requested_sections
        sections = requested_sections(question)
        if intent_name:
            sections = intent_vector_sections(intent_name) or sections
        groups = [list(search_documents(question, embeddings_model, EMBEDDING_MODEL_ID,
                                        drug, section=section, limit=10, **similarity_kwargs))
                  for section in sections]
        records = []
        # Give each requested section space before applying the context budget.
        while any(groups):
            for group in groups:
                if group:
                    records.append(group.pop(0))
        if not records:
            # Relax exactly one level: drop the field filter, never the drug filter.
            records = list(search_documents(question, embeddings_model, EMBEDDING_MODEL_ID,
                                            drug, limit=10, **similarity_kwargs))
            if sections:
                records = [dict(record, metadata={**record['metadata'], 'relaxed': True})
                           for record in records]
        if intent_name and records:
            records = apply_relevance_floor(intent_name, question, drug, records)
        return records
    def generate(system, payload):
        budget = float(os.getenv('ANSWER_MODEL_TIMEOUT_SECONDS', '300'))
        model = create_chat_model(timeout_seconds=budget, max_tokens=1200)
        model = model.bind(format=ENTAILMENT_SCHEMA if system == ENTAILMENT_SYSTEM else OUTPUT_SCHEMA)
        import time
        started = time.monotonic()
        parts = []
        stream = model.stream([('system', system), ('human', payload)])
        try:
            for chunk in stream:
                if time.monotonic() - started > budget:
                    raise TimeoutError('Answer generation exceeded its response budget')
                content = chunk.content
                if isinstance(content, list):
                    parts.extend(block.get('text','') for block in content if isinstance(block,dict))
                else:
                    parts.append(content)
        finally:
            stream.close()
        return ''.join(parts)

    def retrieve_broad(question, drug):
        return search_documents(question, embeddings_model, EMBEDDING_MODEL_ID, drug, limit=10)

    return vector_answer(evidence_index, query, retrieve_vectors, generate,
                         label_lookup=label_lookup,
                         fallback_retrieve=retrieve_broad, direct_lookup=False,
                         allow_grounded_fallback=True, intent=intent_name)

if __name__ == '__main__':
    while True:
        try:
            question = input('DrugCentral> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {'exit', 'quit'}:
            break
        if question:
            import time
            from chatbot.evaluation_trace import capture_trace
            from chatbot.intent import classify as classify_intent
            from chatbot.query_log import build_record, log_query
            started = time.monotonic()
            with capture_trace() as trace:
                result = invoke_english_only(None, question)
            log_query(build_record(question, trace, time.monotonic() - started,
                                   intent=classify_intent(question)))
            print(result)
