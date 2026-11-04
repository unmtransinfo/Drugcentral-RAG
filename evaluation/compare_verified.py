"""Produce a reviewable baseline/new comparison from completed frozen runs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--sweep',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=json.loads((a.baseline/'summary.json').read_text())
    new=json.loads((a.current/'summary.json').read_text())
    if not base['complete'] or not new['complete']:raise ValueError('Both runs must be complete')
    rows=json.loads((a.current/'checkpoint.json').read_text())['rows']
    def value(agg,key):
        v=agg.get(key)
        if isinstance(v,dict):
            n=v['scored'];v=v['mean']
            return 'unmeasured' if v is None else f'{v:.4f} ({n} scored)'
        return 'unmeasured' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
    lines=['# DrugCentral Bedrock evaluation comparison','',
        'Frozen dataset: 240 questions; questions and reference facts were not edited. Both pipelines used the same source snapshot. All reported quality scores are independent Nova model estimates, not clinical review.','',
        f'Baseline artifacts: `{a.baseline}`. Current artifacts: `{a.current}`. Each directory includes per-question traces, category results, summary and failure details.','',
        '## Aggregate results','', '| Metric | Baseline | Bedrock |','|---|---:|---:|']
    for key in ['questions','retrieval_success','recall_at_5','recall_at_10','recall_at_20','final_evidence_recall','faithfulness','factual_correctness','citation_accuracy','quote_accuracy','unsupported_claim_rate','correct_abstention','abstentions','unnecessary_abstention_proxy','repair_questions','seconds','latency_p95','approximate_cost_usd','approximate_cost_per_1000']:
        lines.append(f'| {key} | {value(base["aggregate"],key)} | {value(new["aggregate"],key)} |')
    lines+=['','The baseline answered 212 questions through generation and 28 through its source-excerpt fallback. Baseline generation/entailment failures are recovered answers, not 28 abstentions. Baseline inference/hosting cost was not instrumented, so a cost saving cannot be claimed. Bedrock estimates exclude database hosting, network traffic, ingestion, and offline judging.','',
        'Recall is a mechanical match of frozen reference facts in correct-drug source text. Retrieval success means at least one such fact was retrieved. The baseline retained a smaller trace scope than the new explicit candidate lists; Recall@k is not a like-for-like corpus relevance experiment. Citation accuracy checks valid source references, while the separate judge assesses support. Quote accuracy checks backend slices against authoritative source text; unscored rows are not silently counted as perfect.','',
        '## Category results','', '| Category | Questions | Baseline correctness | Bedrock correctness | Bedrock abstentions | Baseline mean seconds | Bedrock mean seconds |','|---|---:|---:|---:|---:|---:|---:|']
    for category,c in new['by_category'].items():
        b=base['by_category'][category]
        lines.append(f'| {category} | {c["questions"]} | {value(b,"factual_correctness")} | {value(c,"factual_correctness")} | {c["abstentions"]} | {value(b,"seconds")} | {value(c,"seconds")} |')
    lines+=['','## Remaining failed or unscored questions','']
    failures=[]
    for r in rows.values():
        m=r['metrics'];j=r.get('judgment',{})
        if m['abstained'] or j.get('faithfulness')==0 or j.get('factual_correctness')==0 or 'judgment' not in r:
            failures.append(r)
            codes=sorted({d['status'] for o in r['trace'].get('observations',[]) for d in o.get('claim_validation',[]) if d['status']!='SUPPORTED'})
            lines.extend([f'- **{r["id"]}** ({r["category"]}): {r["question"]}',f'  Status: {r["trace"].get("reason") or "SUPPORTED"}; validation failures: {", ".join(codes) or "none"}; judge: {j.get("notes",r.get("judge_error","unscored"))}',f'  Answer: {r["answer"].replace(chr(10)," ")}'])
    if not failures:lines.append('None identified by this run and its available independent judgments.')
    review_path=a.current/'failure_review.json'
    if review_path.exists():
        lines+=['','### Source and trace review','']
        for ident,review in json.loads(review_path.read_text()).items():
            lines.append(f'- **{ident}**: {review["finding"]} Source excerpt: `{review["source"]}`. Origin: {review["origin"]}.')
        lines+=['','The judge classified both abstentions as correct, but source inspection contradicts that for sulindac. Its reported correct-abstention score is therefore unreliable and must not be used as an acceptance criterion. At least one unnecessary full abstention remains; benzoin also permits a supported partial answer. Original judge outputs are preserved rather than silently overwritten.']
    lines+=['','Final pipeline failure categories: `'+json.dumps(new['aggregate']['failure_categories'],sort_keys=True)+'`. Recovered claim failures remain in `failures.json` even when a supported final answer was produced. All withheld answers with retrieved reference evidence are generation/validation issues rather than missing retrieval.','',
        '## Retrieval budget sweep','', '| Candidates / evidence limit | Questions | Candidate reference recall | Selected reference recall |','|---|---:|---:|---:|']
    sweep=json.loads(a.sweep.read_text())
    for key,v in sweep.items():lines.append(f'| {key} | {v["questions"]} | {v["candidate_fact_recall"]:.4f} | {v["selected_fact_recall"]:.4f} |')
    lines+=['','The sweep reused cached embeddings and rerank outputs across budgets. Its timings are not independent cold-start latency comparisons. These predominantly single-field questions often resolve to fewer actual records than any configured budget, so tied recall does not establish that 10/5 is optimal for complex questions.','',
        '## Recommended configuration and limits','',
        'Retain the evaluated 20-candidate / top-8 configuration, equal dense/lexical fusion weights, 1,024-dimensional Cohere embeddings, Qwen3 Next generation and separate entailment, temperature 0, 1,600 output tokens, 60-second model read timeout, and two SDK retries. Full settings and region-specific model IDs are in `environments/prod/bedrock-production.env`. Run it locally with RAG_PIPELINE=verified-bedrock and your own AWS credentials (docs/LOCAL_DEVELOPMENT.md); deployed workloads use their IAM role through the default SDK credential chain.','',
        'The original serving configuration and vector collection are preserved. The verified API is a separate local service backed by live Bedrock; this work does not claim a new publicly hosted AWS deployment. Keep the durable evidence audit enabled. Review the remaining failures before replacing the existing production default. The frozen dataset does not establish clinical safety, performance under load, or abstention sensitivity on genuinely unanswerable questions.','',
        'See `docs/AWS_DEPLOYMENT.md` for architecture and operational commands, `docs/FULL_COVERAGE.md` for migration verification and costs, and `evaluation/DATASETS.md` for the question sets.']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n')
    print(a.output)

if __name__=='__main__':main()
