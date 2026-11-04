"""Create the per-collection search indexes the retriever relies on (idempotent).

Run once after embedding a new vector collection, as the table owner (rag_ingest):
    python -m ingestion.index_collection --collection drugcentral_full_cohere_v4_1024_YYYYMMDD
Index names carry a sha256 prefix of the collection (the same scheme as the existing collection's indexes).
"""
import argparse
from contextlib import closing
import hashlib
import json


def index_statements(collection):
    tag = hashlib.sha256(collection.encode()).hexdigest()[:12]
    where = "WHERE collection = %s"
    return [
        (f'dc_scope_{tag}', f"CREATE INDEX IF NOT EXISTS dc_scope_{tag} ON public.drugcentral_chunks "
                            f"(collection, (metadata->>'drug_name'), (metadata->>'section')) {where}"),
        # Entity documents (targets, diseases, references, ...) have no drug, so section needs its own entry point.
        (f'dc_section_{tag}', f"CREATE INDEX IF NOT EXISTS dc_section_{tag} ON public.drugcentral_chunks "
                              f"(collection, (metadata->>'section')) {where}"),
        (f'dc_lexical_{tag}', f"CREATE INDEX IF NOT EXISTS dc_lexical_{tag} ON public.drugcentral_chunks "
                              f"USING gin (to_tsvector('simple', document)) {where}"),
    ]


def create_indexes(cursor, collection):
    cursor.execute('SELECT 1 FROM public.drugcentral_chunks WHERE collection = %s LIMIT 1', (collection,))
    if not cursor.fetchone():
        raise ValueError('Collection has no chunks; embed it first')
    created = []
    for name, statement in index_statements(collection):
        # The collection name is a bound parameter; CREATE INDEX needs it inline, so it is quoted by the driver.
        cursor.execute(cursor.mogrify(statement, (collection,)).decode())
        created.append(name)
    cursor.execute('ANALYZE public.drugcentral_chunks')
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    from chatbot.storage import _connect
    with closing(_connect()) as conn, conn, conn.cursor() as cur:
        print(json.dumps({'collection': args.collection, 'indexes': create_indexes(cur, args.collection)}, indent=2))


if __name__ == '__main__':
    main()
