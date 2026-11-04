"""Metrics for the complex-question benchmark (retrieval, answer, multi-part and operational).

Inputs are the benchmark questions plus one result row per question:
    {"id", "answer", "seconds", "trace": {route, evidence, observations}}
as written by evaluation/run_complex_benchmark.py. Component credit is derived from the
claims that survived validation and the evidence they cite, not from wording in the answer.
Faithfulness and factual correctness need a judge; pass its verdicts with --judgements.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

KS = (1, 5, 10, 20)
LABEL_ALIASES = {'adverse_events': {'adverse_events', 'adverse_reactions'}}
NOT_ANSWERED_ROUTES = {'abstain'}


def normalize(text):
    return ' '.join(str(text).casefold().split())


def sections_for(section):
    return LABEL_ALIASES.get(section, {section})


def final_metrics(trace):
    merged = {}
    for observation in trace.get('observations', []):
        merged.update(observation.get('metrics', {}))
    return merged


def candidates(trace):
    for observation in reversed(trace.get('observations', [])):
        if 'retrieved_candidates' in observation:
            return observation['retrieved_candidates']
    return trace.get('evidence', [])


def cited_evidence(trace):
    """Evidence cited by claims that passed validation."""
    by_id = {e.get('evidence_id'): e for e in trace.get('evidence', []) if e.get('evidence_id')}
    cited = []
    for observation in trace.get('observations', []):
        for claim in observation.get('validated_claims', []):
            cited.extend(by_id[i] for i in claim.get('evidence_ids', []) if i in by_id)
    return cited


def record_drug(record):
    return normalize(record.get('drug') or record.get('metadata', {}).get('drug_name') or '')


def record_section(record):
    return record.get('section') or record.get('metadata', {}).get('section')


def relevant(record, comp):
    """A record is relevant to a component when it concerns the drug, the field and a gold fact."""
    if comp['drug'] and record_drug(record) != normalize(comp['drug']):
        return False
    if record_section(record) not in sections_for(comp['section']):
        return False
    facts = comp.get('reference_facts') or []
    text = normalize(record.get('text', ''))
    return not facts or any(normalize(f) in text for f in facts)


def retrieval_metrics(records, comp):
    """Recall@k, reciprocal rank and nDCG@10 with binary relevance for one component."""
    flags = [relevant(r, comp) for r in records]
    total_relevant = sum(flags)
    result = {}
    for k in KS:
        result[f'recall@{k}'] = float(any(flags[:k]))
    first = next((i for i, f in enumerate(flags, 1) if f), None)
    result['mrr'] = 1 / first if first else 0.0
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(flags[:10], 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(total_relevant, 10) + 1))
    result['ndcg@10'] = dcg / ideal if ideal else 0.0
    return result


def component_outcome(comp, row):
    trace = row.get('trace', {})
    answer = normalize(row.get('answer', ''))
    cited = cited_evidence(trace)
    answered = any(relevant_scope(e, comp) for e in cited)
    if comp['section'] == 'relationship':
        # Relationship components are graded as a set: how much of the gold set the answer names.
        gold = comp.get('reference_facts') or []
        hits = sum(normalize(g) in answer for g in gold)
        return {'answered': trace.get('route') == 'graph' and hits > 0, 'correct': bool(gold) and hits / len(gold) >= 0.5,
                'set_recall': hits / len(gold) if gold else None}
    facts = comp.get('reference_facts') or []
    correct = answered and (not facts or any(normalize(f) in answer for f in facts))
    return {'answered': answered, 'correct': bool(correct)}


def relevant_scope(record, comp):
    if comp['drug'] and record_drug(record) != normalize(comp['drug']):
        return False
    return record_section(record) in sections_for(comp['section'])


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def summarize(questions, rows, judgements=None):
    by_id = {r['id']: r for r in rows}
    judgements = judgements or {}
    retrieval = defaultdict(list); comp_stats = defaultdict(list); question_stats = defaultdict(list)
    seconds = []; calls = []; costs = []; claims_total = claims_rejected = 0
    citation_ok = citation_total = quote_ok = quote_total = 0
    per_category = defaultdict(lambda: defaultdict(list))
    for q in questions:
        row = by_id.get(q['id'])
        if row is None:
            continue
        trace = row.get('trace', {}); metrics = final_metrics(trace)
        abstained = trace.get('route') in NOT_ANSWERED_ROUTES or bool(metrics.get('abstained'))
        outcomes = [(c, component_outcome(c, row)) for c in q['components']]
        answerable = [(c, o) for c, o in outcomes if c['answerable']]
        unanswerable = [(c, o) for c, o in outcomes if not c['answerable']]
        records = candidates(trace)
        for c, _ in answerable:
            if c['section'] in {'unsupported', 'relationship'}:
                continue
            for name, value in retrieval_metrics([r for r in records if record_drug(r) == normalize(c['drug'] or '')], c).items():
                retrieval[name].append(value)
        for _, o in answerable:
            comp_stats['completed'].append(o['answered'])
            comp_stats['omitted'].append(not o['answered'])
            if o.get('set_recall') is not None:
                comp_stats['set_recall'].append(o['set_recall'])
        if answerable:
            question_stats['all_components_correct'].append(all(o['correct'] for _, o in answerable))
            question_stats['full_abstention'].append(abstained)
        if q['answerable'] == 'partial':
            # Right behaviour: answer the supported parts and do not assert the unsupported ones.
            asserted = any(o['answered'] for _, o in unanswerable)
            question_stats['partial_accuracy'].append(all(o['answered'] for _, o in answerable) and not asserted)
        if q['answerable'] == 'none':
            question_stats['unanswerable_abstained'].append(abstained or not any(o['answered'] for _, o in outcomes))
        seconds.append(row.get('seconds') or metrics.get('total_seconds') or 0)
        bedrock = sum(metrics.get(k, 0) for k in ('embedding_calls', 'rerank_calls', 'generation_calls', 'validation_calls', 'repair_calls'))
        calls.append(bedrock)
        cost = (metrics.get('cost_estimate_usd') or {}).get('total')
        if cost is not None:
            costs.append(cost)
        question_stats['repaired'].append(bool(metrics.get('repair_count')))
        question_stats['abstained'].append(abstained)
        for observation in trace.get('observations', []):
            details = list(observation.get('claim_validation', []))
            for comp in observation.get('components', []):
                details += comp.get('claim_validation', [])
            claims_total += len(details); claims_rejected += sum(d.get('status') != 'SUPPORTED' for d in details)
        authoritative = {e.get('evidence_id'): e for e in trace.get('evidence', []) if e.get('evidence_id')}
        for observation in trace.get('observations', []):
            for claim in observation.get('validated_claims', []):
                citation_total += 1
                citation_ok += all(i in authoritative and authoritative[i].get('source_id') for i in claim.get('evidence_ids', []))
                for quote in claim.get('quotes', []):
                    quote_total += 1
                    text = authoritative.get(quote.get('evidence_id'), {}).get('text', '')
                    quote_ok += bool(quote.get('quote')) and text[quote.get('start', 0):quote.get('end', 0)] == quote['quote']
        cat = per_category[q['category']]
        cat['all_components_correct'].append(all(o['correct'] for _, o in answerable) if answerable else None)
        cat['completion'].append(mean([o['answered'] for _, o in answerable]))
    verdicts = [judgements[q['id']] for q in questions if q['id'] in judgements]
    summary = {
        'questions': len(rows),
        'retrieval': {k: mean(v) for k, v in sorted(retrieval.items())},
        'component_completion_rate': mean(comp_stats['completed']),
        'all_components_correct_rate': mean(question_stats['all_components_correct']),
        'unnecessary_component_omission_rate': mean(comp_stats['omitted']),
        'partial_answer_accuracy': mean(question_stats['partial_accuracy']),
        'unnecessary_full_abstention_rate': mean(question_stats['full_abstention']),
        'unanswerable_abstention_rate': mean(question_stats['unanswerable_abstained']),
        'multi_hop_set_recall': mean(comp_stats['set_recall']),
        'unsupported_claim_rate': claims_rejected / claims_total if claims_total else None,
        'citation_accuracy': citation_ok / citation_total if citation_total else None,
        'exact_quote_accuracy': quote_ok / quote_total if quote_total else None,
        'faithfulness': mean([v.get('faithful') for v in verdicts]),
        'factual_correctness': mean([v.get('correct') for v in verdicts]),
        'latency_p50_seconds': statistics.median(seconds) if seconds else None,
        'latency_p95_seconds': percentile(seconds, 0.95),
        'bedrock_calls_per_question': mean(calls),
        'repair_rate': mean(question_stats['repaired']),
        'abstention_rate': mean(question_stats['abstained']),
        'cost_per_question_usd': mean(costs),
        'cost_per_1000_questions_usd': mean(costs) * 1000 if costs else None,
        'by_category': {c: {k: mean(v) for k, v in m.items()} for c, m in sorted(per_category.items())},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='JSONL from run_complex_benchmark.py')
    parser.add_argument('--judgements', type=Path, help='JSON object: question id -> {"faithful": bool, "correct": bool}')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    judgements = json.loads(args.judgements.read_text()) if args.judgements else None
    summary = summarize(questions, rows, judgements)
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
