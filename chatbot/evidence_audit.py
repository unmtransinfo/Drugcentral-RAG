"""Durable claim-to-source audit without recording raw user questions or drafts."""
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4


def persist(answer,trace):
    path=os.getenv('EVIDENCE_AUDIT_PATH','')
    if not path:
        if os.getenv('AUDIT_REQUIRED')=='1':raise OSError('Evidence audit is required')
        return
    claims=[];metrics={};failures=[]
    for obs in trace.get('observations',[]):
        claims.extend({k:c[k] for k in ('claim_id','text','evidence_ids') if k in c}
                      for c in obs.get('validated_claims',[]))
        metrics.update(obs.get('metrics',{}))
        failures.extend({'claim_id':c.get('claim_id'),'status':c.get('status')}
                        for c in obs.get('claim_validation',[]))
    used={i for c in claims for i in c.get('evidence_ids',[])}
    sources=[]
    for e in trace.get('evidence',[]):
        if e.get('evidence_id') not in used:continue
        m=e.get('metadata',{})
        sources.append({k:e.get(k) for k in ('evidence_id','source_id','text','drug','section','url')} |
                       {'metadata':{k:m[k] for k in ('dataset_version','source_type','product_name','route','form') if k in m}})
    row={'request_id':str(uuid4()),'time':time.time(),'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
         'route':trace.get('route'),'status':trace.get('reason') or 'SUPPORTED','claims':claims,
         'sources':sources,'validation_statuses':failures,'metrics':metrics}
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    # Linux deployment: lock across API worker processes as well as threads.
    import fcntl
    with target.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        handle.flush()
        if os.getenv('AUDIT_REQUIRED')=='1':os.fsync(handle.fileno())
        fcntl.flock(handle,fcntl.LOCK_UN)
    logging.getLogger('uvicorn.error').info('rag_audit request_id=%s answer_sha256=%s status=%s',
        row['request_id'],row['answer_sha256'],row['status'])
