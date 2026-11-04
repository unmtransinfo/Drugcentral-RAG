"""Run the complex benchmark through the live pipeline and write per-question traces.

WARNING: every question calls Bedrock (embedding, rerank, generation, validation, repair). A
multi-part question makes one set of calls per component, so this costs real money. Use --limit
for a small first run and --estimate to see the call budget without calling anything.

Run: python -m evaluation.run_complex_benchmark --questions evaluation/complex-110/questions.json \
        --output benchmarks/complex-110-results.jsonl [--limit 10] [--estimate]
Then: python -m evaluation.complex_metrics --questions ... --results ...
"""
import argparse
import json
from pathlib import Path
import time

from chatbot.components import plan_question

# Upper bound of Bedrock calls per component: embed + rerank + generate + verify + repair + verify.
CALLS_PER_COMPONENT = 6


def estimate(questions):
    total = 0
    for q in questions:
        plan = plan_question(q['question'], [d for d in q.get('drugs', []) if d])
        total += max(1, len(plan['components'])) * CALLS_PER_COMPONENT if plan['type'] == 'MULTI_PART' else CALLS_PER_COMPONENT
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--estimate', action='store_true', help='print the Bedrock call upper bound and exit')
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())[:args.limit]
    if args.estimate:
        print(json.dumps({'questions': len(questions), 'bedrock_calls_upper_bound': estimate(questions)}))
        return
    if args.output.exists():
        raise SystemExit('Refusing to overwrite ' + str(args.output))
    from agent import invoke_english_only
    from chatbot.evaluation_trace import capture_trace
    with args.output.open('w') as out:
        for q in questions:
            started = time.monotonic()
            with capture_trace() as trace:
                answer = invoke_english_only(None, q['question'])
            row = {'id': q['id'], 'answer': answer, 'seconds': time.monotonic() - started,
                   'trace': {k: trace[k] for k in ('route', 'reason', 'evidence', 'observations')}}
            out.write(json.dumps(row, default=str) + '\n'); out.flush()
            print(q['id'], trace['route'], f"{row['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
