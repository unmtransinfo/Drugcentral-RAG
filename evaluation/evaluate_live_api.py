"""Collect fresh answers and source evidence from the active API for independent review.

Reuses frozen questions for comparison. Never sends reference facts to the API,
never substitutes the direct source lookup, and never assigns unreviewed accuracy.

The target API must already be running with ENABLE_EVALUATION_API=true and API_AUTH_TOKEN set;
the same token must be in this process's environment (it is never written to outputs):
    python -m evaluation.evaluate_live_api --url http://127.0.0.1:8000 --output benchmarks/<run>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from evaluation.api_collection import ApiCollector, evidence_documents, retrieval_hit


class ReviewBenchmark(ApiCollector):
    def save(self):
        super().save()
        worksheet = []
        completed = []
        for question in self.questions:
            row = self.state['rows'].get(question['id'], {})
            documents = evidence_documents(row.get('trace', {}))
            # Judge context contains source records only, not generated claims or
            # the model's own validation assertions stored in trace observations.
            documents = list({json.dumps(d, sort_keys=True): d for d in documents}.values())
            status = 'collected' if row.get('collected') else ('error' if row.get('api_error') else 'pending')
            hit = retrieval_hit(question, documents) if status == 'collected' else None
            if status == 'collected':
                completed.append(dict(row, retrieval_success=hit))
            worksheet.append(dict(question, answer=row.get('answer'),
                context=json.dumps(documents, ensure_ascii=False, indent=2) if documents else None,
                collection_status=status, api_error=row.get('api_error'), route=row.get('route'),
                abstain_reason=(row.get('trace') or {}).get('reason'),
                automated_retrieval_success=hit, automated_faithfulness=None,
                automated_correctness=None, human_retrieval_success=None,
                human_faithfulness=None, human_correctness=None, review_notes=''))
        retrieval_scores=[r['retrieval_success'] for r in completed if r['retrieval_success'] is not None]
        table_coverage={}
        view_coverage={}
        for q in self.questions:
            for table in q.get('source_tables',[]):
                item=table_coverage.setdefault(table,{'questions':0,'collected':0})
                item['questions']+=1
                item['collected']+=bool(self.state['rows'].get(q['id'],{}).get('collected'))
        for q in self.questions:
            for view in q.get('source_views',[]):
                item=view_coverage.setdefault(view,{'questions':0,'collected':0})
                item['questions']+=1
                item['collected']+=bool(self.state['rows'].get(q['id'],{}).get('collected'))
        summary = dict(questions=len(self.questions), collected=len(completed),
            pending=len(self.questions)-len(completed), review_status='pending independent review',
            scope=f'Current API vector-RAG pipeline answers (see api_base_url for target); frozen {len(self.questions)}-question set. No independent accuracy judgments yet.',
            table_coverage=table_coverage, view_coverage=view_coverage,
            source_object_count=len(table_coverage)+len(view_coverage),
            retrieval_scored=len(retrieval_scores),
            retrieval_definition='Full-table questions remain unscored without exact table/row provenance in API traces. For legacy drug questions, mechanical check: at least one reference fact occurs in retrieved source text for the correct drug. Includes selected evidence and retrieved candidates presented to the LLM. Does not measure answer correctness or full multi-field coverage.',
            metrics_percent=dict(retrieval_success=100*sum(retrieval_scores)/len(retrieval_scores) if len(completed)==len(self.questions) and retrieval_scores else None,
                faithfulness=None, factual_correctness=None),
            routes=dict(Counter(r.get('route', 'unknown') for r in completed)),
            abstain_reasons=dict(Counter((r.get('trace') or {}).get('reason') or 'unspecified'
                                         for r in completed if r.get('route') == 'abstain')),
            config=self.config)
        for name, value in [('human_review.json', worksheet), ('summary.json', summary)]:
            target=self.path/name
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(target)


def run(args):
    load_dotenv(ROOT/'.env')
    if args.config: load_dotenv(args.config, override=True)
    token=os.environ.get('API_AUTH_TOKEN')
    if not token:
        raise SystemExit('API_AUTH_TOKEN is required; no token is saved in outputs.')
    import requests
    response=requests.get(args.url.rstrip('/')+'/api/evaluation/info',headers={'Authorization':'Bearer '+token},timeout=15)
    response.raise_for_status()
    metadata=response.json()
    if metadata.get('answer_mode')!='vector-rag':
        raise SystemExit('The target API is not running vector-rag; refusing a misleading benchmark.')
    config=dict(api_base_url=args.url, chatbot=metadata,
        questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['agent.py','api.py','chatbot/production_pipeline.py','chatbot/storage.py','chatbot/model_config.py', 'evaluation/evaluate_live_api.py']})
    runner=ReviewBenchmark(json.loads(args.questions.read_text()),config,args.output,token)
    runner.collect()
    print((args.output/'summary.json').read_text(), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT/'evaluation/fresh-240-20260928/questions.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--config', type=Path, help='Optional .env-style file loaded into this process (for example API_AUTH_TOKEN)')
    args=parser.parse_args()
    return run(args)


if __name__=='__main__':
    main()
