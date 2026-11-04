"""Embedding/collection identity utilities.

Commands:
  check    Compare the configured embedding identity with the collection's
           stamped metadata; exit 1 on any mismatch (same check startup runs).
  stamp    Backfill identity metadata for an existing collection; refuses when
           the stored per-chunk embedding model differs from the configuration.
  selftest Self-retrieval smoke test: embed N stored chunks through the query
           path and confirm each returns itself at rank 1.
"""
import argparse
import os
import sys
from contextlib import closing


def _pgvector_only():
    from chatbot.storage import backend_name
    backend_name()  # raises for any VECTOR_BACKEND other than pgvector
    return True


def check():
    if not _pgvector_only():
        return 0
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    from chatbot.storage import _collection, _check_embedding_identity
    from chatbot.database_config import connect_database
    model = embedding_model_id()
    embeddings = create_embeddings(timeout_seconds=60)
    with closing(connect_database()) as conn, conn, conn.cursor() as cursor:
        try:
            _check_embedding_identity(cursor, model, embeddings)
        except ValueError as exc:
            print(f'FAIL: {exc}')
            return 1
    print(f"OK: collection '{_collection()}' matches the configured embedding identity ({model})")
    return 0


def stamp():
    if not _pgvector_only():
        return 0
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    from chatbot.storage import _collection, _stamp_collection
    from chatbot.database_config import connect_database
    model = embedding_model_id()
    with closing(connect_database()) as conn, conn, conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT embedding_model FROM public.drugcentral_chunks "
                       "WHERE collection = %s", (_collection(),))
        identities = {row[0] for row in cursor.fetchall()}
        if identities != {model}:
            print(f"FAIL: stored embedding identities {sorted(identities)} do not match the "
                  f"configured model {model!r}; refusing to stamp. Fix the configuration first.")
            return 1
        cursor.execute("SELECT vector_dims(embedding) FROM public.drugcentral_chunks "
                       "WHERE collection = %s LIMIT 1", (_collection(),))
        row = cursor.fetchone()
        if row is None:
            print(f"FAIL: collection '{_collection()}' is empty")
            return 1
        embeddings = create_embeddings(timeout_seconds=60)
        _stamp_collection(cursor, model, row[0], embeddings)
    print(f"stamped collection '{_collection()}': model={model} dimensions={row[0]}")
    return 0


def reembed(batch_size):
    """Re-materialize the configured collection's vectors with the configured model.

    Reads stored documents and rewrites the same collection in a single
    transaction via store_documents, which also re-stamps the identity. Use
    after an embedding-model rebuild makes stored vectors irreproducible.
    """
    if not _pgvector_only():
        return 0
    from langchain_core.documents import Document
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    from chatbot.storage import _collection, store_documents
    from chatbot.database_config import connect_database
    model = embedding_model_id()
    with closing(connect_database()) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT document, metadata FROM public.drugcentral_chunks "
                       "WHERE collection = %s AND embedding_model = %s ORDER BY chunk_index",
                       (_collection(), model))
        rows = cursor.fetchall()
    if not rows:
        print(f"FAIL: no documents found in collection '{_collection()}' for model {model!r}")
        return 1
    print(f"re-embedding {len(rows)} documents in '{_collection()}' with {model} ...", flush=True)
    documents = [Document(page_content=text, metadata=metadata) for text, metadata in rows]
    embeddings = create_embeddings(timeout_seconds=120)
    store_documents(documents, embeddings, model, batch_size=batch_size)
    print('re-embedding committed and collection re-stamped')
    return 0


def selftest(count):
    if not _pgvector_only():
        return 0
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    from chatbot.storage import _collection, search_documents
    from chatbot.database_config import connect_database
    model = embedding_model_id()
    embeddings = create_embeddings(timeout_seconds=60)
    with closing(connect_database()) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT chunk_index, document FROM public.drugcentral_chunks "
                       "WHERE collection = %s AND embedding_model = %s "
                       "ORDER BY chunk_index LIMIT %s", (_collection(), model, count))
        rows = cursor.fetchall()
    if not rows:
        print(f"FAIL: no stored chunks for model {model!r} in '{_collection()}'")
        return 1
    failures = 0
    for chunk_index, document in rows:
        results = search_documents(document, embeddings, model, None, limit=10, raw=True)
        top = results[0]['text'] if results else None
        ok = top == document
        failures += 0 if ok else 1
        print(f"chunk {chunk_index}: {'OK' if ok else 'FAIL (self not rank 1)'}")
    print(f"self-retrieval: {len(rows) - failures}/{len(rows)} passed")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'stamp', 'selftest', 'reembed'])
    parser.add_argument('--n', type=int, default=3, help='selftest probe count')
    parser.add_argument('--batch-size', type=int, default=100, help='reembed batch size')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if args.command == 'check':
        return check()
    if args.command == 'stamp':
        return stamp()
    if args.command == 'reembed':
        return reembed(args.batch_size)
    return selftest(args.n)


if __name__ == '__main__':
    sys.exit(main())
