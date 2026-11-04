"""Add collection-scoped lexical and optional ANN indexes without replacing data."""
import argparse
import hashlib
import os
from contextlib import closing


def build(collection,dimensions,ann=False):
    from psycopg2 import sql
    from chatbot.storage import _connect
    if dimensions not in {256,512,1024,1536}: raise ValueError('Invalid dimension')
    suffix=hashlib.sha256(collection.encode()).hexdigest()[:12]
    with closing(_connect()) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            for name,expression,method in [
                ('scope',sql.SQL("collection,(metadata->>'drug_name'),(metadata->>'section')"),'btree'),
                ('lexical',sql.SQL("to_tsvector('simple',document)"),'gin')]:
                index_name='dc_'+name+'_'+suffix
                cur.execute('SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(%s)',('public.'+index_name,))
                existing=cur.fetchone()
                if existing is not None and not existing[0]:
                    cur.execute(sql.SQL('DROP INDEX CONCURRENTLY {}').format(sql.Identifier(index_name)))
                cur.execute(sql.SQL('CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON public.drugcentral_chunks USING {} ({}) WHERE collection={}').format(
                    sql.Identifier('dc_'+name+'_'+suffix),sql.SQL(method),expression,sql.Literal(collection)))
            if ann:
                cur.execute(sql.SQL('CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON public.drugcentral_chunks USING hnsw ((embedding::vector({})) vector_cosine_ops) WHERE collection={}').format(
                    sql.Identifier('dc_dense_'+suffix),sql.Literal(dimensions),sql.Literal(collection)))
            cur.execute('ANALYZE public.drugcentral_chunks')


def main():
    from dotenv import load_dotenv
    load_dotenv()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',required=True)
    p.add_argument('--dimensions',type=int,default=1024);p.add_argument('--ann',action='store_true')
    a=p.parse_args();build(a.collection,a.dimensions,a.ann)

if __name__=='__main__':main()
