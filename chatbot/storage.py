"""Storage selection shared by ingestion and the agent.

The agent uses metadata lookups, so reads return documents without vectors.
PostgreSQL stores embeddings in pgvector columns for future similarity queries.
"""

import json
import os
import re
from contextlib import closing
from chatbot.embedding_config import embedding_backend


def backend_name():
    name = os.getenv("VECTOR_BACKEND", "pgvector").strip().lower()
    if name != "pgvector":
        raise ValueError(f"VECTOR_BACKEND={name!r} is not supported; only pgvector is available")
    return name


def _connect():
    from chatbot.database_config import connect_database
    return connect_database()


def _collection():
    return os.getenv("VECTOR_COLLECTION", "drugcentral")

# Fixed sentinel embedded at ingestion/stamp time and re-embedded at startup;
# the pair must match, which catches silent model drift that name/dimension
# checks cannot (a rebuilt model producing different vectors under the same name).
PROBE_TEXT = 'DrugCentral embedding identity probe v1'


def _cosine_distance(first, second):
    dot = sum(x * y for x, y in zip(first, second))
    norm_a = sum(x * x for x in first) ** 0.5
    norm_b = sum(x * x for x in second) ** 0.5
    return 1.0 - dot / (norm_a * norm_b)


def _decode_vector(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def _ensure_collections_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS public.drugcentral_collections (
            collection text PRIMARY KEY,
            backend text NOT NULL,
            model text NOT NULL,
            profile text NOT NULL DEFAULT '',
            dimensions integer NOT NULL,
            doc_prefix text NOT NULL DEFAULT '',
            probe_text text,
            probe_vector jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    # Tables created before probe columns existed are upgraded in place.
    cursor.execute("ALTER TABLE public.drugcentral_collections ADD COLUMN IF NOT EXISTS probe_text text")
    cursor.execute("ALTER TABLE public.drugcentral_collections ADD COLUMN IF NOT EXISTS probe_vector jsonb")


def _stamp_collection(cursor, model, dimensions, embeddings=None):
    """Record the embedding identity produced by ingestion for later startup checks."""
    _ensure_collections_table(cursor)
    probe_vector = None
    if embeddings is not None:
        probe_vector = json.dumps(embeddings.embed_documents([PROBE_TEXT])[0], allow_nan=False)
    cursor.execute(
        "INSERT INTO public.drugcentral_collections "
        "(collection, backend, model, profile, dimensions, doc_prefix, probe_text, probe_vector) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
        "ON CONFLICT (collection) DO UPDATE SET backend=EXCLUDED.backend, model=EXCLUDED.model, "
        "profile=EXCLUDED.profile, dimensions=EXCLUDED.dimensions, doc_prefix=EXCLUDED.doc_prefix, "
        "probe_text=EXCLUDED.probe_text, probe_vector=EXCLUDED.probe_vector",
        (_collection(), embedding_backend(), model,
         '',
         dimensions, os.getenv('EMBED_DOC_PREFIX', ''),
         PROBE_TEXT if probe_vector else None, probe_vector),
    )


def _check_embedding_identity(cursor, model, embeddings=None, *, read_only=False):
    """Refuse to serve a collection whose vectors differ from the configuration.

    A dimension comparison alone is insufficient (different models can share
    768 dimensions), so backend, model, profile and dimensions are all checked.
    When a probe vector is stamped, the configured embedder must reproduce it
    (cosine distance < 1e-3), which catches silent model rebuilds. Legacy
    unstamped collections are verified against one stored chunk and then
    stamped once by maintenance callers. Runtime reads validate without writing;
    a legacy collection that the configured embedder cannot
    reproduce is rejected with a re-ingestion instruction.
    """
    if read_only:
        cursor.execute("SELECT to_regclass('public.drugcentral_collections')")
        metadata_exists = cursor.fetchone()[0] is not None
    else:
        _ensure_collections_table(cursor)
        metadata_exists = True
    cursor.execute("SELECT chunk_index, document, embedding::text FROM public.drugcentral_chunks "
                   "WHERE collection = %s ORDER BY chunk_index LIMIT 1", (_collection(),))
    row = cursor.fetchone()
    stored_dims = None
    sample = None
    if row is not None:
        sample = (row[1], _decode_vector(row[2]))
        stored_dims = len(sample[1])
    stamp = None
    if metadata_exists:
        # JSON projection also supports older metadata tables without probe columns.
        cursor.execute("SELECT backend, model, profile, dimensions, "
                       "to_jsonb(c)->'probe_vector' AS probe_vector "
                       "FROM public.drugcentral_collections c WHERE collection = %s", (_collection(),))
        stamp = cursor.fetchone()
    if stamp is None:
        if stored_dims is not None:
            if embeddings is not None and sample is not None:
                fresh = embeddings.embed_documents([sample[0]])[0]
                if len(fresh) != len(sample[1]) or _cosine_distance(fresh, sample[1]) > 1e-3:
                    raise ValueError(
                        f"Collection '{_collection()}' is not stamped and the configured "
                        f"embedder does not reproduce a stored vector (cosine distance above "
                        f"1e-3): its embeddings were built by a different model build. "
                        f"Re-ingest the collection (python -m chatbot.embedding_check reembed) "
                        f"or restore the original embedding model.")
            if read_only and embeddings is None:
                raise ValueError("An embedder is required to validate an unstamped collection")
            if not read_only:
                _stamp_collection(cursor, model, stored_dims, embeddings)
        return
    backend, stamped_model, profile, dimensions, probe_vector = stamp
    configured_profile = ''
    problems = []
    if backend != embedding_backend():
        problems.append(f"backend stamped '{backend}' but configured '{embedding_backend()}'")
    if stamped_model != model:
        problems.append(f"model stamped '{stamped_model}' but configured '{model}'")
    if profile != configured_profile:
        problems.append(f"profile stamped '{profile}' but configured '{configured_profile}'")
    if stored_dims is not None and dimensions != stored_dims:
        problems.append(f"stamped {dimensions} dimensions but stored vectors have {stored_dims}")
    if probe_vector and embeddings is not None:
        fresh = embeddings.embed_documents([PROBE_TEXT])[0]
        if _cosine_distance(fresh, _decode_vector(probe_vector)) > 1e-3:
            problems.append("the configured embedder does not reproduce the stamped probe vector "
                            "(the embedding model was rebuilt or replaced since ingestion)")
    if problems:
        raise ValueError(
            f"Embedding collection identity mismatch for collection '{_collection()}': "
            + "; ".join(problems)
            + ". Fix RAG_PIPELINE/EMBED_BACKEND, the embedding model (OLLAMA_EMBED_MODEL or BEDROCK_EMBED_MODEL/"
            "BEDROCK_EMBED_DIMENSIONS) or VECTOR_COLLECTION, or re-ingest the collection.")


def load_documents(embeddings, model):
    structured = os.getenv("STRUCTURED_COLLECTION")
    if structured and backend_name() == "pgvector":
        with closing(_connect()) as conn, conn, conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT embedding_model FROM public.drugcentral_chunks WHERE collection = %s", (_collection(),))
            identities = {row[0] for row in cursor.fetchall()}
            if identities != {model}:
                raise ValueError("Vector collection is empty or embedding identity mismatches the selected backend")
            _check_embedding_identity(cursor, model, embeddings, read_only=True)
            cursor.execute("SELECT document, metadata FROM public.drugcentral_evidence "
                           "WHERE collection = %s ORDER BY chunk_index", (structured,))
            rows = cursor.fetchall()
        if not rows:
            raise ValueError("Structured collection is empty")
        return {"documents": [row[0] for row in rows], "metadatas": [row[1] for row in rows]}
    with closing(_connect()) as conn, conn, conn.cursor() as cursor:
        _check_embedding_identity(cursor, model, embeddings, read_only=True)
        cursor.execute(
            "SELECT document, metadata, embedding_model FROM public.drugcentral_chunks "
            "WHERE collection = %s ORDER BY chunk_index", (_collection(),)
        )
        rows = cursor.fetchall()
    if not rows:
        raise ValueError("pgvector collection is empty; run python -m ingestion.vectors first")
    if any(row[2] != model for row in rows):
        raise ValueError("Stored embedding identity differs from the configured embedding backend/model")
    return {"documents": [r[0] for r in rows], "metadatas": [r[1] for r in rows]}


def _refuse_identity_change(cursor, model):
    """Never overwrite a collection built by another embedding backend or model (for example Bedrock vs Ollama)."""
    cursor.execute("SELECT to_regclass('public.drugcentral_collections')")
    if cursor.fetchone()[0] is None:
        return
    cursor.execute("SELECT backend, model FROM public.drugcentral_collections WHERE collection = %s", (_collection(),))
    stamp = cursor.fetchone()
    if stamp is not None and tuple(stamp) != (embedding_backend(), model):
        raise ValueError(
            f"Collection '{_collection()}' holds {stamp[0]} vectors from '{stamp[1]}'; the configured embedder is "
            f"{embedding_backend()} '{model}'. Choose a different VECTOR_COLLECTION; vectors from different "
            "embedding models are never mixed or overwritten.")


def store_documents(documents, embeddings, model, batch_size=100):
    """Build a collection. PostgreSQL replacement commits only on full success."""
    if not documents:
        raise ValueError("Refusing to ingest an empty dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    from psycopg2.extras import Json, execute_values

    with closing(_connect()) as conn, conn, conn.cursor() as cursor:
        # The extension must be provisioned by the database administrator.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.drugcentral_chunks (
                collection text NOT NULL,
                chunk_index integer NOT NULL,
                document text NOT NULL,
                metadata jsonb NOT NULL,
                embedding_model text NOT NULL,
                embedding vector NOT NULL,
                PRIMARY KEY (collection, chunk_index)
            )
        """)
        # Serialize rebuilds; readers retain the previous committed snapshot.
        cursor.execute("LOCK TABLE public.drugcentral_chunks IN SHARE ROW EXCLUSIVE MODE")
        if os.getenv("INGEST_REQUIRE_NEW_COLLECTION") == "1":
            cursor.execute("SELECT 1 FROM public.drugcentral_chunks WHERE collection=%s LIMIT 1", (_collection(),))
            if cursor.fetchone():
                raise ValueError("Collection already exists; select a NEW collection")
        _refuse_identity_change(cursor, model)
        cursor.execute("DELETE FROM public.drugcentral_chunks WHERE collection = %s", (_collection(),))
        dimensions = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors = embeddings.embed_documents([doc.page_content for doc in batch])
            if len(vectors) != len(batch):
                raise ValueError("Embedding count does not match document count")
            for vector in vectors:
                dimensions = dimensions or len(vector)
                if not vector or len(vector) != dimensions:
                    raise ValueError("Embedding dimensions must be nonzero and consistent")
            execute_values(
                cursor,
                "INSERT INTO public.drugcentral_chunks "
                "(collection, chunk_index, document, metadata, embedding_model, embedding) VALUES %s",
                [(_collection(), i + j, doc.page_content, Json(doc.metadata), model,
                  json.dumps(vector, allow_nan=False))
                 for j, (doc, vector) in enumerate(zip(batch, vectors))],
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )
            print(f"Stored {min(i + batch_size, len(documents))}/{len(documents)} chunks (uncommitted)")
        if dimensions is None:
            raise ValueError("No embeddings were produced; refusing to stamp the collection")
        _stamp_collection(cursor, model, dimensions, embeddings)


def _rank_candidates(candidates, query, limit, raw=False):
    """Keep similarity order within section-priority groups; remove empty records."""
    if raw:
        return candidates[:limit]
    from chatbot.retrieval import EMPTY, SECTION_HEADERS, section_blocks
    from chatbot.question_retrieval import requested_sections
    requested = set(requested_sections(query))
    useful = []
    for candidate in candidates:
        blocks = list(section_blocks(candidate["text"]))
        meaningful = [(header, body) for header, body in blocks if any(
            re.sub(r"^[-*•]\s*", "", line.strip()).strip().rstrip(".;").lower() not in EMPTY
            for line in body.splitlines()
            if re.sub(r"^[-*•]\s*", "", line.strip()).strip().rstrip(".;"))]
        if not meaningful:
            continue
        sections = {candidate["metadata"].get("section")}
        sections.update(SECTION_HEADERS.get(header.rsplit(" — ", 1)[-1]) for header, _ in meaningful)
        useful.append((bool(requested & sections), candidate))
    # Python's stable sort preserves vector ranking inside each priority group.
    useful.sort(key=lambda item: not item[0])
    return [candidate for _, candidate in useful[:limit]]


def search_documents(query, embeddings, model, drug_name, section=None, limit=4, raw=False,
                     min_similarity=None):
    """Drug-scoped similarity, excluding empty records and prioritizing named sections.

    raw=True exposes unmodified vector order for embedding evaluation. No data is
    rewritten. Rank all candidates for this drug before limiting, so placeholders
    cannot crowd valid evidence out of the candidate window.

    min_similarity (cosine-distance models only) drops candidates whose cosine
    similarity to the query is below the floor; dropped candidates simply never
    become records, so callers see them as absent evidence.
    """
    if not 1 <= limit <= 10:
        raise ValueError("A limit between 1 and 10 is required")
    distance_operator = "<=>"
    vector = json.dumps(embeddings.embed_query(query), allow_nan=False)
    max_distance = None if min_similarity is None else 1.0 - float(min_similarity)
    with closing(_connect()) as conn, conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT document, metadata FROM public.drugcentral_chunks "
            "WHERE collection = %s AND embedding_model = %s "
            "AND (%s IS NULL OR metadata->>'drug_name' = %s) "
            "AND (%s IS NULL OR metadata->>'section' = %s OR NOT metadata ? 'section') "
            "AND (%s::float8 IS NULL OR embedding <=> %s::vector <= %s::float8) "
            f"ORDER BY embedding {distance_operator} %s::vector, chunk_index LIMIT %s",
            (_collection(), model, drug_name, drug_name, section, section,
             max_distance, vector, max_distance, vector, max(limit, 40)),
        )
        candidates = [{"text": row[0], "metadata": row[1]} for row in cursor.fetchall()]
    return _rank_candidates(candidates, query, limit, raw)


def publish_structured_snapshot(records, collection):
    """Publish a new immutable evidence snapshot without recomputing embeddings.

    Vector collections are retained separately for semantic candidate discovery.
    The caller must select this snapshot explicitly with STRUCTURED_COLLECTION.
    """
    from psycopg2.extras import Json, execute_values
    if not collection or not records:
        raise ValueError("A named, nonempty snapshot is required")
    if any(record["metadata"].get("schema_version") != 2 for record in records):
        raise ValueError("Only schema version 2 records can be published")
    with closing(_connect()) as conn, conn, conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.drugcentral_evidence (
                collection text NOT NULL,
                chunk_index integer NOT NULL,
                document text NOT NULL,
                metadata jsonb NOT NULL,
                PRIMARY KEY (collection, chunk_index)
            )
        """)
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("drugcentral-evidence:" + collection,))
        cursor.execute("SELECT 1 FROM public.drugcentral_evidence WHERE collection = %s LIMIT 1", (collection,))
        if cursor.fetchone():
            raise ValueError("Snapshot already exists; choose a new collection name")
        for start in range(0, len(records), 1000):
            execute_values(cursor, "INSERT INTO public.drugcentral_evidence VALUES %s",
                           [(collection, start + offset, record["text"], Json(record["metadata"]))
                            for offset, record in enumerate(records[start:start + 1000])])
