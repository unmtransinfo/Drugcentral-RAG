"""Independent pgvector and full-text retrieval, then reciprocal rank fusion."""
import hashlib
import json
import math
import os
import re
import time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from chatbot.storage import _connect, _collection
from chatbot.retrieval import NAME_STOPWORDS
from chatbot.question_retrieval import requested_sections


def limits():
    candidates=int(os.getenv('RETRIEVAL_CANDIDATES','20'))
    top=int(os.getenv('RERANK_TOP_K','8'))
    if not 1<=top<=candidates<=100:
        raise ValueError('Require 1 <= RERANK_TOP_K <= RETRIEVAL_CANDIDATES <= 100')
    return candidates,top


def record_key(record):
    return hashlib.sha256(json.dumps([record['metadata'].get('source_id'),record['metadata'].get('source_url'),record['metadata'].get('drug_name'),record['text']],sort_keys=True).encode()).hexdigest()


def fuse(groups,limit):
    merged={}; scores={}
    weights={'dense':float(os.getenv('DENSE_WEIGHT','1')),'lexical':float(os.getenv('LEXICAL_WEIGHT','1'))}
    if any(not math.isfinite(w) or w<0 for w in weights.values()) or sum(weights.values())==0:
        raise ValueError('Retrieval weights must be nonnegative and not both zero')
    for kind,records in groups:
        seen=set()
        for rank,record in enumerate(records,1):
            key=record_key(record)
            if key in seen: continue
            seen.add(key)
            merged.setdefault(key,dict(record))
            merged[key].update({k:v for k,v in record.items() if k.endswith('_score')})
            scores[key]=scores.get(key,0)+weights[kind]/(60+rank)
    return [dict(merged[k],fusion_score=scores[k]) for k in sorted(merged,key=lambda k:(-scores[k],k))[:limit]]


def lexical_query(query):
    """tsquery text for the exact-word search; empty when the question has no usable word.

    PostgreSQL indexes a hyphenated code such as U-1101 or KW-0002 as the two tokens 'u' and '-1101' (the minus sign stays
    attached to the number), so a query built only from 'u' and '1101' can never match the number. The signed token is added
    so an exact code outranks the thousands of other codes that share the prefix. Digits only, so nothing can inject syntax.
    """
    text=query.lower()
    terms=sorted(set(re.findall(r'[a-z0-9]+',text))-NAME_STOPWORDS)
    terms+=[f"'-{digits}'" for digits in sorted(set(re.findall(r'(?<=[a-z0-9])-(\d+)',text)))]
    return ' | '.join(terms)


def _search(query,vector,model,drug,section,limit,kind):
    filters="collection=%s AND embedding_model=%s AND (%s IS NULL OR metadata->>'drug_name'=%s) AND (%s IS NULL OR metadata->>'section'=%s)"
    params=[_collection(),model,drug,drug,section,section]
    if kind=='dense':
        sql=f"SELECT document,metadata,chunk_index,1-(embedding <=> %s::vector) FROM public.drugcentral_chunks WHERE {filters} ORDER BY embedding <=> %s::vector,chunk_index LIMIT %s"
        params=[vector]+params+[vector,max(40,limit*2)]
    else:
        tsquery=lexical_query(query)
        if not tsquery: return []
        expression="to_tsvector('simple', document)"
        sql=f"SELECT document,metadata,chunk_index,ts_rank_cd({expression},to_tsquery('simple',%s)) FROM public.drugcentral_chunks WHERE {filters} AND {expression} @@ to_tsquery('simple',%s) ORDER BY 4 DESC,chunk_index LIMIT %s"
        params=[tsquery]+params+[tsquery,max(40,limit*2)]
    with closing(_connect()) as conn,conn,conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '15000'")
        cur.execute(sql,params)
        rows=cur.fetchall()
    records=[dict(text=text,metadata={**meta,'source_id':meta.get('source_id') or f'pgvector:{_collection()}:{chunk}',
        'source_url':meta.get('source_url') or (f"https://drugcentral.org/drugcard/{meta['struct_id']}" if meta.get('struct_id') is not None else None)},
        **{('semantic_score' if kind=='dense' else 'lexical_score'):float(score)}) for text,meta,chunk,score in rows]
    from chatbot.retrieval_ranking import evidence_sections
    return [r for r in records if evidence_sections(r)]


def hybrid_retrieve(question,drugs,embeddings,model,metrics):
    count,_=limits()
    sections=list(dict.fromkeys(requested_sections(question))) or [None]
    requests=[(drug,section) for drug in drugs or [None] for section in sections]
    started=time.monotonic()
    vector=json.dumps(embeddings.embed_query(question),allow_nan=False)
    metrics['embedding_seconds']=time.monotonic()-started
    groups=[]
    def run(kind,drug,section):
        started=time.monotonic()
        records=_search(question,vector,model,drug,section,count,kind)
        return kind,records,time.monotonic()-started
    with ThreadPoolExecutor(max_workers=min(8,len(requests)*2)) as pool:
        pending=[pool.submit(run,kind,drug,section) for drug,section in requests for kind in ('dense','lexical')]
        for future in pending:
            kind,records,seconds=future.result()
            metrics[kind+'_retrieval_seconds']=metrics.get(kind+'_retrieval_seconds',0)+seconds
            groups.append((kind,records))
    ranked=fuse(groups,count*len(requests))
    representatives=[]
    for drug,section in requests:
        best=next((r for r in ranked if (drug is None or r['metadata'].get('drug_name')==drug)
                   and (section is None or r['metadata'].get('section')==section)),None)
        if best is not None: representatives.append(best)
    seen=set(); selected=[]
    for r in representatives+ranked:
        key=record_key(r)
        if key not in seen:
            seen.add(key); selected.append(r)
    metrics['retrieval_subqueries']=len(requests)
    return selected[:count]
