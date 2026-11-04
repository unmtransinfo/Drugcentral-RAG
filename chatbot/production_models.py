"""Bedrock adapters; all identity, region, and generation controls are configuration."""
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from threading import RLock, local
from functools import lru_cache


def setting(name, default=None):
    value = os.getenv(name, default)
    if value is None or value == '':
        raise ValueError(f'{name} is required')
    return value


def validate_settings():
    for name in ('AWS_REGION','BEDROCK_EMBED_MODEL','BEDROCK_RERANK_MODEL','BEDROCK_GENERATION_MODEL'):
        setting(name)
    for name,default,low,high in [('REQUEST_TIMEOUT','60',1,300),('MAX_RETRIES','2',0,5),
            ('GENERATION_MAX_TOKENS','1600',1,8192),('GENERATION_TEMPERATURE','0',0,1),
            ('GENERATION_TOP_P','1',0.0001,1),('EMBED_CACHE_SIZE','512',0,10000),
            ('RERANK_CACHE_SIZE','256',0,10000)]:
        value=float(setting(name,default))
        if not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'Invalid {name}')
    for name,value in os.environ.items():
        if name.startswith('COST_') and value:
            number=float(value)
            if not math.isfinite(number) or number<0:raise ValueError(f'Invalid {name}')


def embed_identity():
    return f"bedrock:{setting('BEDROCK_EMBED_MODEL')}:dim={int(setting('BEDROCK_EMBED_DIMENSIONS','1024'))}:document=search_document:query=search_query:float:truncate=NONE"


@lru_cache(maxsize=8)
def client(region):
    from chatbot.runtime_mode import require_package
    boto3 = require_package('boto3', 'Bedrock mode', 'environments/aws/requirements-aws.lock')
    from botocore.config import Config
    return boto3.Session().client('bedrock-runtime', region_name=region, config=Config(
        connect_timeout=5, read_timeout=float(setting('REQUEST_TIMEOUT','60')),
        retries={'mode':'standard','total_max_attempts':1 + int(setting('MAX_RETRIES','2'))}))


class CohereEmbeddings:
    def __init__(self, runtime=None):
        self.model = setting('BEDROCK_EMBED_MODEL')
        self.dimensions = int(setting('BEDROCK_EMBED_DIMENSIONS','1024'))
        if self.dimensions not in {256,512,1024,1536}:
            raise ValueError('Unsupported Cohere Embed v4 dimension')
        self.runtime = runtime or client(setting('BEDROCK_EMBED_REGION', setting('AWS_REGION','us-east-2')))
        self.identity = embed_identity()
        self.cache = OrderedDict()
        self.lock = RLock()
        self.request_usage = local()

    def _embed(self, texts, input_type):
        if not texts:
            return []
        if any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError('Embedding text must be nonempty')
        result=[]
        for start in range(0,len(texts),48):
            batch=texts[start:start+48]
            response=self.runtime.invoke_model(modelId=self.model, contentType='application/json',
                accept='application/json', body=json.dumps({'texts':batch,'input_type':input_type,
                    'embedding_types':['float'],'output_dimension':self.dimensions,'truncate':'NONE'}))
            body=response['body']
            try:
                vectors=json.loads(body.read())['embeddings']['float']
            finally:
                body.close()
            if len(vectors)!=len(batch) or any(len(v)!=self.dimensions or
                    not all(type(x) in (int,float) and math.isfinite(x) for x in v) or
                    not any(v) for v in vectors):
                raise ValueError('Invalid Cohere embedding response')
            result.extend(vectors)
        return result

    def embed_documents(self,texts):
        return self._embed(texts,'search_document')

    def embed_query(self,text):
        # Immutable embeddings depend on model/config, not on mutable source contents.
        key=hashlib.sha256((self.identity+'\0'+text).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.request_usage.estimated_tokens=0
                return list(self.cache[key])
        vector=self._embed([text],'search_query')[0]
        self.request_usage.estimated_tokens=(len(text)+3)//4
        with self.lock:
            self.cache[key]=tuple(vector)
            while len(self.cache)>int(setting('EMBED_CACHE_SIZE','512')):
                self.cache.popitem(last=False)
        return vector


_RANK_CACHE=OrderedDict()
_RANK_LOCK=RLock()
_RANK_USAGE=local()


def rerank(question, records, runtime=None):
    if not records:
        return []
    region=setting('BEDROCK_RERANK_REGION',setting('AWS_REGION','us-east-2'))
    key=hashlib.sha256(json.dumps([question,records,setting('BEDROCK_RERANK_MODEL'),region,setting('RERANK_TOP_K','8')],sort_keys=True).encode()).hexdigest()
    with _RANK_LOCK:
        cached=_RANK_CACHE.get(key) if runtime is None else None
        if cached is not None:
            _RANK_CACHE.move_to_end(key)
            _RANK_USAGE.calls=0
            return [dict(records[i],rerank_score=score) for i,score in cached]
    _RANK_USAGE.calls=1
    response=(runtime or client(region)).invoke_model(modelId=setting('BEDROCK_RERANK_MODEL'),
        contentType='application/json',accept='application/json',body=json.dumps({
            'api_version':2,'query':question,'documents':[
                json.dumps({'drug':r['metadata'].get('drug_name'),'section':r['metadata'].get('section'),
                            'text':r['text']},ensure_ascii=False) for r in records],
            'top_n':len(records)}))
    body=response['body']
    try:
        results=json.loads(body.read())['results']
    finally:
        body.close()
    out=[]; seen=set()
    for r in results:
        i=r['index']; score=r['relevance_score']
        if type(i) is not int or not 0<=i<len(records) or i in seen or not math.isfinite(score):
            raise ValueError('Invalid reranker response')
        seen.add(i)
        out.append({**records[i], 'rerank_score':score})
    if len(out)!=len(records):
        raise ValueError('Incomplete rerank response')
    results=select_components(question,records,results,int(setting('RERANK_TOP_K','8')))
    out=[dict(records[r['index']],rerank_score=r['relevance_score']) for r in results]
    if runtime is None:
        with _RANK_LOCK:
            _RANK_CACHE[key]=tuple((r['index'],r['relevance_score']) for r in results)
            while len(_RANK_CACHE)>int(setting('RERANK_CACHE_SIZE','256')):
                _RANK_CACHE.popitem(last=False)
    return out


def select_components(question,records,ranked,limit):
    """Retain the highest reranked member of each requested drug/field scope."""
    from chatbot.question_retrieval import requested_sections
    sections=requested_sections(question)
    drugs=list(dict.fromkeys(r['metadata'].get('drug_name') for r in records))
    if len(drugs)>4:drugs=[None]
    selected=[];seen=set()
    for drug in drugs:
        for section in sections:
            match=next((r for r in ranked if (drug is None or records[r['index']]['metadata'].get('drug_name')==drug)
                and records[r['index']]['metadata'].get('section') in
                    ({'adverse_events','adverse_reactions'} if section=='adverse_events' else {section})),None)
            if match is not None and match['index'] not in seen:
                selected.append(match);seen.add(match['index'])
    for r in ranked:
        if r['index'] not in seen:selected.append(r);seen.add(r['index'])
    chosen={r['index'] for r in selected[:limit]}
    # Preserve the model's order within the coverage-constrained final set.
    return [r for r in ranked if r['index'] in chosen]


class IncompleteResponse(ValueError):
    """The model stopped for a reason other than end_turn; `reason` is the Bedrock stopReason."""
    def __init__(self,message,reason=None):
        super().__init__(message); self.reason=reason


class TruncatedResponse(IncompleteResponse):
    """The output hit the maxTokens cap, so the JSON object is cut off."""


class MalformedResponse(ValueError):
    """The model finished but its text is not a leading JSON object."""


def converse(system,payload,stage,metrics,runtime=None):
    model=setting('BEDROCK_VALIDATION_MODEL',setting('BEDROCK_GENERATION_MODEL')) if stage=='validation' else setting('BEDROCK_GENERATION_MODEL')
    region=setting('AWS_REGION','us-east-2')
    started=time.monotonic()
    metrics[stage+'_calls']=metrics.get(stage+'_calls',0)+1
    try:
        response=(runtime or client(region)).converse(modelId=model,
            system=[{'text':system}],messages=[{'role':'user','content':[{'text':json.dumps(payload,ensure_ascii=False)}]}],
            inferenceConfig={'maxTokens':int(setting('GENERATION_MAX_TOKENS','1600')),
                             'temperature':float(setting('GENERATION_TEMPERATURE','0')),
                             'topP':float(setting('GENERATION_TOP_P','1'))})
    finally:
        metrics[stage+'_seconds']=metrics.get(stage+'_seconds',0)+time.monotonic()-started
    usage=response.get('usage',{})
    for key in ['inputTokens','outputTokens']:
        metrics[stage+'_'+key]=metrics.get(stage+'_'+key,0)+usage.get(key,0)
    stop=response.get('stopReason')
    metrics[stage+'_stop_reason']=stop
    if stop=='max_tokens':
        metrics[stage+'_truncated']=metrics.get(stage+'_truncated',0)+1
        raise TruncatedResponse('Model output reached the token limit',stop)
    if stop!='end_turn':
        raise IncompleteResponse('Incomplete model response',stop)
    raw=''.join(b['text'] for b in response['output']['message']['content'] if 'text' in b)
    # Some models wrap otherwise valid JSON in a code fence.
    raw=raw.strip()
    if raw.startswith('```json') and raw.endswith('```'):
        raw=raw[7:-3].strip()
    # Only a leading JSON object can supply claims/verdicts. Extra commentary
    # or duplicate objects are discarded, never rendered or used as evidence.
    # Incomplete output still fails above; malformed first JSON still fails here.
    try:
        value,end=json.JSONDecoder().raw_decode(raw)
    except ValueError as exc:
        raise MalformedResponse('Model output is not valid JSON') from exc
    if not isinstance(value,dict):
        raise MalformedResponse('Expected a structured object')
    if raw[end:].strip():
        metrics[stage+'_discarded_suffix']=True
    return value
