"""Resume an isolated Cohere collection build; never overwrite a serving collection."""
import argparse
import hashlib
import json
import os
import random
import time
from threading import Lock
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Transient Bedrock conditions: wait and retry. Anything else (validation, access, quota exhaustion) fails immediately.
RETRYABLE={'ThrottlingException','TooManyRequestsException','InternalServerException','ServiceUnavailableException',
           'ModelTimeoutException','ModelNotReadyException'}


def embed_with_retry(embedder,texts,attempts=8,sleep=time.sleep):
    """Embed one batch, backing off on transient service errors (about 5 minutes in total before giving up)."""
    for attempt in range(attempts):
        try: return embedder.embed_documents(texts)
        except Exception as exc:
            code=(getattr(exc,'response',None) or {}).get('Error',{}).get('Code')
            if code not in RETRYABLE or attempt==attempts-1: raise
            sleep(min(60,5*2**attempt)+random.random())


def migrate(source,target,apply=False,reuse=None):
    from chatbot.storage import _connect, _stamp_collection
    from chatbot.production_models import CohereEmbeddings, embed_identity
    from psycopg2.extras import Json, execute_values
    if target==os.getenv('VECTOR_COLLECTION') or target==source:
        raise ValueError('Select a new target distinct from serving and source collections')
    model=embed_identity()
    with closing(_connect()) as read,read,read.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.drugcentral_evidence WHERE collection=%s',(source,))
        count=cur.fetchone()[0]
    if not count: raise ValueError('Empty source snapshot')
    report=dict(source=source,target=target,count=count,model=model,dimensions=int(os.getenv('BEDROCK_EMBED_DIMENSIONS','1024')))
    if not apply: return report
    embedder=CohereEmbeddings()
    reusable=None
    if reuse:
        # Copy vectors for chunks whose text is byte-identical in an already-complete collection built with the SAME model;
        # only changed or new chunks are embedded (and paid for).
        with closing(_connect()) as check,check,check.cursor() as cur:
            cur.execute('SELECT model FROM public.drugcentral_collections WHERE collection=%s',(reuse,))
            row=cur.fetchone()
            if not row or row[0]!=model: raise ValueError('The reuse collection is missing or was built with a different embedding identity')
            cur.execute('SELECT status FROM public.drugcentral_migrations WHERE collection=%s',(reuse,))
            row=cur.fetchone()
            if not row or row[0]!='complete': raise ValueError('The reuse collection must be a complete migration')
            tag=hashlib.sha256(reuse.encode()).hexdigest()[:12]
            cur.execute(cur.mogrify('CREATE INDEX IF NOT EXISTS dc_md5_'+tag+' ON public.drugcentral_chunks (md5(document)) WHERE collection = %s',(reuse,)).decode())
        reusable=reuse
    throttle_lock=Lock()
    next_request=[0.0]
    def embed_batch(texts):
        # Conservatively estimate biomedical token density; keep headroom for queries.
        delay=sum(len(t) for t in texts)/float(os.getenv('INGEST_CHARS_PER_TOKEN','4'))/max(1,int(os.getenv('INGEST_TOKENS_PER_MINUTE','240000')))*60
        with throttle_lock:
            wait=max(0,next_request[0]-time.monotonic())
            next_request[0]=max(next_request[0],time.monotonic())+delay
        if wait: time.sleep(wait)
        return embed_with_retry(embedder,texts)
    with closing(_connect()) as write,closing(_connect()) as read:
        with write,write.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(hashtext(%s))',('cohere-migration:'+target,))
            if not cur.fetchone()[0]: raise ValueError('Migration already running')
            cur.execute('SELECT 1 FROM public.drugcentral_collections WHERE collection=%s',(target,))
            if cur.fetchone(): raise ValueError('Completed collection already exists; choose a new target')
            cur.execute('CREATE TABLE IF NOT EXISTS public.drugcentral_migrations (collection text PRIMARY KEY, model text NOT NULL, source text NOT NULL, source_count bigint NOT NULL, status text NOT NULL, source_sha256 text)')
            cur.execute('SELECT model,source,source_count FROM public.drugcentral_migrations WHERE collection=%s',(target,))
            existing=cur.fetchone()
            if existing and existing!=(model,source,count): raise ValueError('Migration identity changed')
            if not existing:
                cur.execute('SELECT 1 FROM public.drugcentral_chunks WHERE collection=%s LIMIT 1',(target,))
                if cur.fetchone(): raise ValueError('Unrelated target already exists')
                cur.execute('INSERT INTO public.drugcentral_migrations VALUES (%s,%s,%s,%s,%s,NULL)',(target,model,source,count,'building'))
        checksum=hashlib.sha256(); completed=0
        with read,read.cursor(name='cohere_source') as source_cur, ThreadPoolExecutor(max_workers=2) as pool:
            source_cur.execute('SELECT chunk_index,document,metadata FROM public.drugcentral_evidence WHERE collection=%s ORDER BY chunk_index',(source,))
            while True:
                # Four 48-document requests in flight; persist each group for safe resume.
                rows=source_cur.fetchmany(192)
                if not rows: break
                for idx,text,meta in rows:
                    checksum.update(json.dumps([idx,text,meta],sort_keys=True,ensure_ascii=False).encode())
                with write,write.cursor() as cur:
                    cur.execute('SELECT chunk_index,document,metadata,embedding_model FROM public.drugcentral_chunks WHERE collection=%s AND chunk_index=ANY(%s)',(target,[r[0] for r in rows]))
                    stored={r[0]:r[1:] for r in cur.fetchall()}
                pending=[]
                for idx,text,meta in rows:
                    meta={**meta,'source_id':f'pg-evidence:{source}:{idx}','source_collection':source,
                          'source_chunk_index':idx}
                    # Entity documents (targets, diseases, ...) have no drug card; do not invent a link.
                    url=meta.get('source_url') or (f"https://drugcentral.org/drugcard/{meta['struct_id']}" if meta.get('struct_id') is not None else None)
                    if url: meta['source_url']=url
                    if idx in stored:
                        if stored[idx]!=(text,meta,model): raise ValueError('Source changed during migration; choose a new target')
                    else: pending.append((idx,text,meta))
                copied=0
                if reusable and pending:
                    digests={hashlib.md5(t.encode()).hexdigest():(i,t,m) for i,t,m in pending}
                    with write,write.cursor() as cur:
                        cur.execute('SELECT DISTINCT ON (md5(document)) md5(document), embedding::text FROM public.drugcentral_chunks '
                                    'WHERE collection=%s AND md5(document)=ANY(%s)',(reusable,list(digests)))
                        found={d:v for d,v in cur.fetchall()}
                        hits=[(idx,text,meta,found[hashlib.md5(text.encode()).hexdigest()]) for idx,text,meta in pending
                              if hashlib.md5(text.encode()).hexdigest() in found]
                        if hits:
                            execute_values(cur,'INSERT INTO public.drugcentral_chunks (collection,chunk_index,document,metadata,embedding_model,embedding) VALUES %s',
                                [(target,idx,text,Json(meta),model,vec) for idx,text,meta,vec in hits],template='(%s,%s,%s,%s,%s,%s::vector)')
                    copied=len(hits); done={h[0] for h in hits}
                    pending=[r for r in pending if r[0] not in done]
                batches=[pending[i:i+48] for i in range(0,len(pending),48)]
                futures=[pool.submit(embed_batch,[r[1] for r in batch]) for batch in batches]
                for batch,future in zip(batches,futures):
                    vectors=future.result()
                    with write,write.cursor() as cur:
                        execute_values(cur,'INSERT INTO public.drugcentral_chunks (collection,chunk_index,document,metadata,embedding_model,embedding) VALUES %s',
                            [(target,idx,text,Json(meta),model,json.dumps(vector,allow_nan=False)) for (idx,text,meta),vector in zip(batch,vectors)],
                            template='(%s,%s,%s,%s,%s,%s::vector)')
                completed+=len(rows)
                print(json.dumps({'processed':completed,'total':count,'new':len(pending),'reused':copied}),flush=True)
        with write,write.cursor() as cur:
            cur.execute('SELECT count(*),min(vector_dims(embedding)),max(vector_dims(embedding)) FROM public.drugcentral_chunks WHERE collection=%s',(target,))
            if cur.fetchone()!=(count,embedder.dimensions,embedder.dimensions): raise ValueError('Incomplete target')
            cur.execute('SELECT count(*) FROM public.drugcentral_chunks c LEFT JOIN public.drugcentral_evidence e '
                        'ON e.collection=%s AND e.chunk_index=c.chunk_index WHERE c.collection=%s '
                        'AND (e.chunk_index IS NULL OR c.document<>e.document OR NOT c.metadata @> e.metadata)',(source,target))
            if cur.fetchone()[0]:raise ValueError('Source text/metadata changed during migration')
            # Snapshot IDs/text/metadata were compared on resume; final marker gates serving.
            previous=os.environ.get('VECTOR_COLLECTION'); os.environ['VECTOR_COLLECTION']=target
            try: _stamp_collection(cur,model,embedder.dimensions,embedder)
            finally:
                if previous is None: os.environ.pop('VECTOR_COLLECTION',None)
                else: os.environ['VECTOR_COLLECTION']=previous
            cur.execute("UPDATE public.drugcentral_migrations SET status='complete',source_sha256=%s WHERE collection=%s",(checksum.hexdigest(),target))
        report.update(status='complete',source_sha256=checksum.hexdigest())
    return report


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--target',required=True)
    parser.add_argument('--apply',action='store_true');parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--reuse',help='complete collection (same model) whose vectors are copied for unchanged chunk text')
    args=parser.parse_args()
    report=migrate(args.source,args.target,args.apply,args.reuse)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
