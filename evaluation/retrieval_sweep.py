"""Compare candidate/evidence budgets on frozen questions; no answer generation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from evaluation.verified_benchmark import fact_recall


def main():
    from dotenv import load_dotenv
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',type=Path,default=Path('evaluation/fresh-240-20260928/questions.json'))
    p.add_argument('--config',default='environments/prod/bedrock-production.env')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dense-weight',type=float)
    p.add_argument('--lexical-weight',type=float)
    p.add_argument('--candidates',type=int,nargs='+',default=[10,20,30,40])
    p.add_argument('--top-k',type=int,nargs='+',default=[5,8,10])
    a=p.parse_args();load_dotenv('.env');load_dotenv(a.config,override=True)
    if a.dense_weight is not None:os.environ['DENSE_WEIGHT']=str(a.dense_weight)
    if a.lexical_weight is not None:os.environ['LEXICAL_WEIGHT']=str(a.lexical_weight)
    from agent import evidence_index,embeddings_model,EMBEDDING_MODEL_ID
    from chatbot.drug_resolution import resolve_question
    from chatbot.hybrid_search import hybrid_retrieve
    from chatbot.production_models import rerank,select_components
    questions=json.loads(a.questions.read_text());rows=[]
    manifest=a.output.with_suffix('.manifest.json')
    signature=hashlib.sha256(json.dumps({'questions':questions,'candidates':a.candidates,'top_k':a.top_k,
        'config':Path(a.config).read_text(),'weights':[os.getenv('DENSE_WEIGHT'),os.getenv('LEXICAL_WEIGHT')],'sources':{str(p):p.read_text() for p in Path('chatbot').glob('*.py')}},sort_keys=True).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if json.loads(manifest.read_text())['signature']!=signature:raise ValueError('Sweep inputs changed; choose a new output')
        rows=json.loads(a.output.read_text())
    else:manifest.write_text(json.dumps({'signature':signature,'concurrency':4})+'\n')
    def one(q,n):
        drugs,clarification=resolve_question(evidence_index,q['question'])
        if clarification:return [{'id':q['id'],'category':q['category'],'candidates':n,'status':'CLARIFICATION'}]
        started=time.monotonic();metrics={}
        candidates=hybrid_retrieve(q['question'],drugs,embeddings_model,EMBEDDING_MODEL_ID,metrics)
        selected=rerank(q['question'],candidates)
        selections={k:[selected[r['index']] for r in select_components(q['question'],selected,
                    [{'index':i,'relevance_score':r['rerank_score']} for i,r in enumerate(selected)],k)] for k in a.top_k if k<=n}
        return [{'id':q['id'],'category':q['category'],'candidates':n,'top_k':k,
            'candidate_fact_recall':fact_recall(q,candidates),'selected_fact_recall':fact_recall(q,selections[k]),
            'seconds':time.monotonic()-started,'candidate_ids':[r['metadata']['source_id'] for r in candidates],
            'selected_ids':[r['metadata']['source_id'] for r in selections[k]]} for k in a.top_k if k<=n]
    for n in a.candidates:
        # Settings stay constant for every worker in this batch.
        os.environ['RETRIEVAL_CANDIDATES']=str(n);os.environ['RERANK_TOP_K']=str(min(max(a.top_k),n))
        done={r['id'] for r in rows if r['candidates']==n}
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(one,q,n):q for q in questions if q['id'] not in done}
            for future in as_completed(jobs):
                rows.extend(future.result())
                temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(a.output)
                print(n,jobs[future]['id'],flush=True)
    grouped=defaultdict(list)
    for row in rows:
        if 'top_k' in row:grouped[f"{row['candidates']}/{row['top_k']}"] .append(row)
    summary={key:{'questions':len(values),'candidate_fact_recall':statistics.mean(r['candidate_fact_recall'] for r in values),
                  'selected_fact_recall':statistics.mean(r['selected_fact_recall'] for r in values),
                  'seconds':statistics.mean(r['seconds'] for r in values)} for key,values in grouped.items()}
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
