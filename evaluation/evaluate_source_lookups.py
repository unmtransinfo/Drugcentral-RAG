"""Regression-check production source lookup answers against a frozen question set.

Uses the full configured source database; reference facts enter scoring only,
never answer generation. These exact-fact checks are not an independent LLM judge.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chatbot.retrieval import EvidenceIndex, source_grounded_answer
from chatbot.evaluation_trace import capture_trace
from evaluation.api_collection import evidence_documents, retrieval_hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'benchmarks/web-judge-240/human_review.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory to preserve existing results')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from chatbot.storage import load_documents, backend_name
    from chatbot.embedding_config import embedding_model_id
    import os
    if backend_name() != 'pgvector' or not os.getenv('STRUCTURED_COLLECTION'):
        raise ValueError('This runner requires the configured full structured pgvector collection')
    print('Loading full configured source snapshot...', flush=True)
    data = load_documents(None, embedding_model_id())
    index = EvidenceIndex(data)
    del data
    print(f'Loaded {len(index.sections)} drugs.', flush=True)
    questions = json.loads(args.questions.read_text())
    rows = []
    worksheet = []
    for q in questions:
        with capture_trace() as trace:
            answer = source_grounded_answer(index, q['question'])
        claims = [line[2:] for line in (answer or '').splitlines() if line.startswith('- ')]
        evidence = evidence_documents(trace)
        supported = [any(d['drug'] == q['drug'] and claim in d['text'] for d in evidence) for claim in claims]
        # All emitted fact bullets must match an acceptable frozen reference.
        correct = bool(claims) and all(claim in q['reference_facts'] for claim in claims)
        row = dict(id=q['id'], category=q['category'], question=q['question'],
                   answer=answer, context=json.dumps({'evidence':trace['evidence'], 'observations':trace['observations']}),
                   trace=trace, route=trace['route'], collected=answer is not None,
                   retrieval_success=retrieval_hit(q, evidence),
                   source_fact_support=sum(supported)/len(supported) if supported else 0.0,
                   exact_reference_correctness=float(correct),
                   faithfulness=None, factual_correctness=None,
                   generation_error=None if answer is not None else 'Not handled by source lookup route')
        rows.append(row)
        worksheet.append(dict(q, answer=answer, context=row['context'],
                              automated_faithfulness=None, automated_correctness=None,
                              human_faithfulness=None, human_correctness=None, review_notes=''))
    metrics = ['retrieval_success', 'source_fact_support', 'exact_reference_correctness']
    def aggregate(items):
        return {key: 100 * sum(r[key] for r in items)/len(items) for key in metrics}
    summary = dict(questions=len(rows), handled=sum(r['collected'] for r in rows),
                   source_drugs=len(index.sections), dataset_versions=sorted(index.versions),
                   scope='Full configured snapshot; production source lookup route only. Mechanical fact-bullet support and exact-reference checks, not independent faithfulness/correctness judging or live API validation.',
                   metrics_percent=aggregate(rows),
                   by_category={cat:aggregate([r for r in rows if r['category']==cat]) for cat in sorted({r['category'] for r in rows})},
                   routes=dict(Counter(r['route'] for r in rows)),
                   failed_ids=[r['id'] for r in rows if not all(r[k] == 1 for k in metrics)],
                   questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                   code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['chatbot.retrieval.py','chatbot.source_queries.py','chatbot.question_retrieval.py']})
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'checkpoint.json').write_text(json.dumps({'config':{'mode':'source-lookup-regression','dataset_versions':sorted(index.versions)},'rows':{r['id']:r for r in rows}},indent=2)+'\n')
    (args.output/'human_review.json').write_text(json.dumps(worksheet,indent=2)+'\n')
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()
