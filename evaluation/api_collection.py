"""Collect answers and evidence traces from a running API through its evaluation jobs (no judging here).

Shared by evaluate_live_api (collection for independent review) and evaluate_source_lookups
(evidence/retrieval scoring helpers). Requires ENABLE_EVALUATION_API=true and API_AUTH_TOKEN on the server.
"""
import hashlib
import json
import time
from pathlib import Path
import requests


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def evidence_documents(trace):
    """Extract returned evidence with drug identity; never infer it from the answer."""
    documents = []
    def walk(value, drug=None):
        if isinstance(value, dict):
            drug = value.get('drug', value.get('drug_name', drug))
            if drug and isinstance(value.get('text'), str):
                documents.append({'drug': drug, 'text': value['text'], 'section': value.get('section'), 'source_url':value.get('url')})
            elif drug and isinstance(value.get('facts'), list):
                documents.append({'drug':drug, 'text': '\n'.join(' '.join(map(str,f)) if isinstance(f,list) else str(f) for f in value['facts'])})
            # Search-by-indication observations use a condition field rather than text.
            elif drug and isinstance(value.get('condition'), str):
                documents.append({'drug':drug, 'text':value['condition']})
            for child in value.values():
                if isinstance(child,(dict,list)):walk(child,drug)
        elif isinstance(value,list):
            for child in value:walk(child,drug)
    walk(trace.get('evidence', []));walk(trace.get('observations', []))
    return documents


def retrieval_hit(question, documents):
    # A matching scalar in an unrelated row is not exact table/record provenance.
    if question.get('benchmark_scope') == 'full_source_table_coverage':
        return None
    return float(any(normalize(d['drug']) == normalize(question['drug']) and
                     any(normalize(fact) in normalize(d['text']) for fact in question['reference_facts'])
                     for d in documents))


class ApiCollector:
    """Resumable collection of API answers into <output>/checkpoint.json."""

    def __init__(self, questions, config, output, api_token):
        self.questions=questions;self.config=config
        self.api_token=api_token  # Intentionally never saved in config/checkpoints.
        self.path=Path(output);self.path.mkdir(parents=True,exist_ok=True)
        if not questions or len({q['id'] for q in questions})!=len(questions):raise ValueError('Invalid question IDs')
        self.signature=digest({'questions':questions,'config':config,'runner':Path(__file__).read_text()})
        self.state_file=self.path/'checkpoint.json'
        self.state={'signature':self.signature,'config':config,'rows':{}}
        if self.state_file.exists():
            self.state=json.loads(self.state_file.read_text())
            if self.state['signature']!=self.signature:raise ValueError('Configuration changed: choose a new output directory')
        self.save()

    def save(self):
        temp=self.state_file.with_suffix('.tmp');temp.write_text(json.dumps(self.state,indent=2));temp.replace(self.state_file)

    def api_request(self, method, path, **kwargs):
        response=requests.request(method,self.config['api_base_url'].rstrip('/')+path,
            headers={'Authorization':'Bearer '+self.api_token},timeout=45,allow_redirects=False,**kwargs)
        if response.is_redirect:raise ValueError('API redirected: use the final HTTPS API URL')
        response.raise_for_status()
        return response.json()

    def collect(self):
        live=self.api_request('GET','/api/evaluation/info')
        if live!=self.config['chatbot']:raise ValueError('Chatbot metadata changed: start a new run')
        for number,q in enumerate(self.questions,1):
            row=self.state['rows'].setdefault(q['id'],{'id':q['id'],'question':q['question'],'category':q['category'],
                 'retrieval_success':None,'faithfulness':None,'factual_correctness':None})
            if row.get('collected'):continue
            print(f'{number}/{len(self.questions)} {q["id"]}',flush=True)
            try:
                if not row.get('job_id'):
                    row['job_id']=self.api_request('POST','/api/evaluation/jobs',json={'question':q['question']})['job_id']
                    self.save()
                deadline=time.monotonic()+1800
                while True:
                    try:
                        job=self.api_request('GET','/api/evaluation/jobs/'+row['job_id'])
                    except requests.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code==404:
                            row.pop('job_id',None)
                        raise
                    if job['status']=='error':
                        row.pop('job_id',None)
                        raise RuntimeError(job['error'])
                    if job['status']=='done':break
                    if time.monotonic()>deadline:raise TimeoutError('Job still running; rerun the command to resume polling')
                    time.sleep(2)
                result=job['result']
                if result['metadata']!=self.config['chatbot']:raise ValueError('Chatbot changed during evaluation')
                trace=result['trace']
                if not trace.get('recorded'):raise ValueError('Missing trace: restart the updated API before evaluating')
                if not isinstance(result['answer'],str) or not result['answer'].strip():raise ValueError('Empty chatbot answer')
                docs=evidence_documents(trace)
                # Both prefetched evidence and selected tool observations were available to the final answer.
                context=json.dumps({'evidence':trace['evidence'],'observations':trace['observations']},ensure_ascii=False,indent=2)
                row.update(answer=result['answer'],trace=trace,context=context,retrieved=docs,
                    retrieval_success=retrieval_hit(q,docs),collected=True,api_error=None,route=trace['route'],
                    generation_error=None,faithfulness=None,factual_correctness=None)
            except Exception as exc:
                row.update(api_error=str(exc),judge_error=None)
                self.save()
                # Do not spend hundreds of failed calls or silently omit failures from the denominator.
                raise RuntimeError(f"Collection stopped at {q['id']}: {exc}. Fix the connection and rerun to resume.") from exc
            self.save()
