"""Frozen-question evaluation with per-question traces, fact recall and explicit unknowns."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def normalize(text): return ' '.join(text.casefold().split())


def source_records(trace,candidates=False):
    if not candidates: return trace.get('evidence',[])
    for obs in reversed(trace.get('observations',[])):
        if 'retrieved_candidates' in obs:
            return obs['retrieved_candidates']
    return trace.get('evidence',[])


def fact_recall(question,records,k=None):
    facts=question.get('reference_facts',[])
    if not facts: return None
    # Table-row questions have no drug; match their reference facts against all retrieved text.
    drug=question.get('drug')
    texts=[normalize(r.get('text','')) for r in records[:k] if not drug or normalize(str(r.get('drug') or r.get('metadata',{}).get('drug_name','')))==normalize(drug)]
    return sum(any(normalize(f) in t for t in texts) for f in facts)/len(facts)


def metrics_for(question,row):
    trace=row['trace']; candidates=source_records(trace,True); evidence=source_records(trace)
    metrics={};claims=[];details=[]
    for obs in trace.get('observations',[]):
        metrics.update(obs.get('metrics',{}));claims+=obs.get('validated_claims',[]);details+=obs.get('claim_validation',[])
    exact=[]; citations=[]
    authoritative={e.get('evidence_id'):e for e in evidence if e.get('evidence_id')}
    for c in claims:
        if c.get('evidence_ids'):
            citations.append(all(i in authoritative and authoritative[i].get('source_id') for i in c['evidence_ids']))
            for q in c.get('quotes',[]):
                e=authoritative.get(q.get('evidence_id'),{})
                exact.append(e.get('text','')[q.get('start',0):q.get('end',0)]==q.get('quote') and bool(q.get('quote')))
        elif type(c.get('evidence_id')) is int:
            # Legacy IDs reference retrieved_candidates, while trace.evidence can
            # contain only the used subset. Do not assume the old positional ID
            # indexes the selected-evidence list.
            quote=c.get('quote','');found=any(quote and quote in e.get('source_text',e.get('text','')) for e in evidence)
            exact.append(found);citations.append(found)
    recall=fact_recall(question,candidates)
    abstained=trace.get('route')=='abstain'
    return {'retrieval_success':None if recall is None else float(recall>0),
        **{f'recall_at_{k}':fact_recall(question,candidates,k) for k in (5,10,20)},
        'final_evidence_recall':fact_recall(question,evidence),
        'quote_accuracy':sum(exact)/len(exact) if exact else None,
        'citation_accuracy':sum(citations)/len(citations) if citations else None,
        'abstained':abstained,'unnecessary_abstention_proxy':bool(abstained and recall and recall>0),
        'faithfulness':row.get('judgment',{}).get('faithfulness'),
        'factual_correctness':row.get('judgment',{}).get('factual_correctness'),
        'unsupported_claim_rate':row.get('judgment',{}).get('unsupported_claim_rate'),
        'correct_abstention':row.get('judgment',{}).get('correct_abstention') if abstained else None,
        'seconds':row.get('seconds',metrics.get('total_seconds')),
        'approximate_cost_usd':metrics.get('cost_estimate_usd',{}).get('total'),
        'known_cost_subtotal_usd':sum(v for k,v in metrics.get('cost_estimate_usd',{}).items() if k!='total' and v is not None) or None,
        'repair_count':metrics.get('repair_count',0),
        'validation_failure_count':metrics.get('validation_failure_count',0),
        'recovered':bool(trace.get('route')=='grounded-fallback' or (metrics.get('repair_count') and not abstained)),
        'failure_status':trace.get('reason'),
        'failure_origin':('retrieval' if not recall else 'validation_or_generation') if (abstained or trace.get('reason')) else None}


def aggregate(rows):
    numeric=['retrieval_success','recall_at_5','recall_at_10','recall_at_20','final_evidence_recall',
             'quote_accuracy','citation_accuracy','faithfulness','factual_correctness','unsupported_claim_rate',
             'correct_abstention','seconds','approximate_cost_usd','known_cost_subtotal_usd']
    result={'questions':len(rows)}
    for key in numeric:
        values=[r['metrics'][key] for r in rows if r['metrics'].get(key) is not None]
        result[key]={'mean':statistics.mean(values) if values else None,'scored':len(values)}
    timings=sorted(r['metrics']['seconds'] for r in rows if r['metrics'].get('seconds') is not None)
    result['latency_p95']=timings[min(len(timings)-1,int(len(timings)*.95))] if timings else None
    result['abstentions']=sum(r['metrics']['abstained'] for r in rows)
    result['unnecessary_abstention_proxy']=sum(r['metrics']['unnecessary_abstention_proxy'] for r in rows)
    result['repair_questions']=sum(bool(r['metrics']['repair_count']) for r in rows)
    result['failure_categories']=dict(Counter(r['metrics']['failure_status'] for r in rows if r['metrics']['failure_status']))
    mean=result['approximate_cost_usd']['mean']
    result['approximate_cost_per_1000']=mean*1000 if mean is not None else None
    return result


def save(output,state,questions):
    rows=[];categories=defaultdict(list)
    for q in questions:
        if q['id'] not in state['rows']: continue
        row=state['rows'][q['id']];row['metrics']=metrics_for(q,row)
        rows.append(row);categories[q['category']].append(row)
    summary={'expected_questions':len(questions),'collected':len(rows),'complete':len(rows)==len(questions),
             'aggregate':aggregate(rows),'by_category':{k:aggregate(v) for k,v in categories.items()},
             'definitions':{'recall_at_k':'Fraction of frozen reference facts appearing verbatim (case/whitespace normalized) in correct-drug candidate text; a mechanical proxy, not expert relevance judgments.',
             'faithfulness':'Separate source-only model judgment when present; not generator confidence or human clinical validation.',
             'unknown':'Missing annotations, token rates or timing remain null, never zero.',
             'unnecessary_abstention_proxy':'Abstention with at least one reference fact retrieved; this does not prove that every requested qualifier was answerable.'},
             'config':state.get('config',{})}
    output.mkdir(parents=True,exist_ok=True)
    failures=[dict(id=r['id'],question=r['question'],category=r['category'],answer=r['answer'],metrics=r['metrics'],trace=r['trace']) for r in rows if r['metrics']['abstained'] or r['metrics']['failure_status'] or r['metrics']['validation_failure_count'] or r['metrics']['factual_correctness']==0 or r['metrics']['faithfulness']==0]
    for name,value in [('checkpoint.json',state),('summary.json',summary),('failures.json',failures)]:
        temp=output/(name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(output/name)


def run(args):
    from dotenv import load_dotenv
    load_dotenv('.env');
    if args.config:load_dotenv(args.config,override=True)
    questions=json.loads(args.questions.read_text())
    output=args.output
    if args.import_checkpoint:
        original=json.loads(args.import_checkpoint.read_text())
        state={'config':original.get('config',{}),'rows':{k:v for k,v in original['rows'].items() if v.get('collected')}}
        # Baseline HTTP logs retained backend latency; no fabricated timing for old runs.
        log=args.import_checkpoint.parent/'api.log'
        if log.exists():
            import re
            durations=[float(x) for x in re.findall(r'chat_completed .*? seconds=(\d+(?:\.\d+)?)',log.read_text())]
            completed=list(state['rows'].values())
            if len(durations)==len(completed):
                for row,seconds in zip(completed,durations):row['seconds']=seconds
        save(output,state,questions);return
    from agent import invoke_english_only
    from chatbot.evaluation_trace import capture_trace
    source_files=['agent.py',*map(str,Path('chatbot').glob('*.py'))]
    config={k:v for k,v in os.environ.items() if k in {'AWS_REGION','BEDROCK_EMBED_MODEL','BEDROCK_RERANK_MODEL','BEDROCK_RERANK_REGION','BEDROCK_GENERATION_MODEL','BEDROCK_VALIDATION_MODEL','VECTOR_COLLECTION','STRUCTURED_COLLECTION','RETRIEVAL_CANDIDATES','RERANK_TOP_K','DENSE_WEIGHT','LEXICAL_WEIGHT','GENERATION_MAX_TOKENS','GENERATION_TEMPERATURE','GENERATION_TOP_P'}}
    signature=hashlib.sha256(json.dumps({'questions':questions,'config':config,'code':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files}},sort_keys=True).encode()).hexdigest()
    state={'signature':signature,'config':config,'rows':{}}
    if (output/'checkpoint.json').exists():
        state=json.loads((output/'checkpoint.json').read_text())
        if state['signature']!=signature:raise ValueError('Changed code or configuration: choose a new output directory')
    for q in questions:
        if q['id'] in state['rows']:continue
        started=time.monotonic()
        with capture_trace() as trace:answer=invoke_english_only(None,q['question'])
        state['rows'][q['id']]={'id':q['id'],'question':q['question'],'category':q['category'],'answer':answer,'trace':trace,'seconds':time.monotonic()-started}
        save(output,state,questions)
        print(json.dumps({'completed':len(state['rows']),'total':len(questions),'id':q['id'],'status':trace.get('reason') or trace['route']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',type=Path,default=Path('evaluation/fresh-240-20260928/questions.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--import-checkpoint',type=Path)
    run(p.parse_args())

if __name__=='__main__':main()
