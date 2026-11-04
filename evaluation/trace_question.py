"""Show what the pipeline did for ONE question: routing, retrieval, what the model was shown, what it returned and why
claims were accepted or rejected. Use it to explain a miss. Makes the same Bedrock calls as a normal question (a few cents).

Run inside the API container (it needs the loaded index, database and Bedrock access):
    docker exec -i ec2-api-1 python -m evaluation.trace_question "What does the Orange Book patent use code U-1101 mean?"
"""
from collections import Counter
import json
import sys

from chatbot.question_retrieval import requested_sections


def head(text, n=150):
    return ' '.join(str(text).split())[:n]


def find_runs(trace):
    """(label, observation dict) for each generation: the single path or one per component."""
    runs = []
    for obs in trace.get('observations', []):
        if 'components' in obs:
            for comp in obs['components']:
                runs.append((f"component {comp.get('id')} [{comp.get('status')}]", comp))
        elif 'generated' in obs or 'claim_validation' in obs:
            runs.append(('single path', obs))
    return runs


def render(question, answer, trace):
    out = [f'QUESTION: {question}', f'ROUTING: sections={requested_sections(question)} route={trace.get("route")} reason={trace.get("reason")}']
    metrics, candidates = {}, []
    for obs in trace.get('observations', []):
        metrics.update(obs.get('metrics', {}))
        if 'retrieved_candidates' in obs:
            candidates = obs['retrieved_candidates']
        if 'plan' in obs:
            out.append('PLAN: ' + json.dumps({'type': obs['plan'].get('type'), 'components': [
                (c.get('drug'), c.get('section')) for c in obs['plan'].get('components', [])]}))
        if 'graph' in obs:
            out.append('GRAPH: ' + json.dumps(obs['graph'])[:300])
    keys = ('retrieved_candidate_count', 'reranked_evidence_count', 'repair_count', 'component_count', 'components_supported')
    out.append('METRICS: ' + ', '.join(f'{k}={metrics[k]}' for k in keys if k in metrics))
    mix = Counter((c.get('drug') or '(entity)', (c.get('metadata') or {}).get('section')) for c in candidates)
    out.append(f'CANDIDATES RETRIEVED ({len(candidates)}): ' + '; '.join(f'{d}/{s} x{n}' for (d, s), n in mix.most_common(8)))
    out.append('EVIDENCE SHOWN TO THE MODEL:')
    for i, e in enumerate(trace.get('evidence', [])):
        out.append(f"  [{i}] {e.get('drug') or '(entity)'} / {e.get('section')}: {head(e.get('text'))}")
    if not trace.get('evidence'):
        out.append('  (none)')
    for label, run in find_runs(trace):
        claims = (run.get('generated') or {}).get('claims') if isinstance(run.get('generated'), dict) else None
        out.append(f'MODEL OUTPUT ({label}): ' + (json.dumps([c.get('text') for c in claims])[:400] if claims else 'no claims returned'))
        for status in run.get('claim_validation', []):
            out.append(f"  validation {status.get('claim_id')}: {status.get('status')}")
    out.append(f'ANSWER: {head(answer, 300)}')
    return '\n'.join(out)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    question = ' '.join(sys.argv[1:])
    from agent import invoke_english_only
    from chatbot.evaluation_trace import capture_trace
    with capture_trace() as trace:
        answer = invoke_english_only(None, question)
    print(render(question, answer, trace))


if __name__ == '__main__':
    main()
