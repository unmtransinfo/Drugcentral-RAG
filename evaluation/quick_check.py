"""Ask every question of a benchmark through the normal /api/chat endpoint and report, per source table, how many were
answered and how many answers contain the expected value.

This is a MECHANICAL check, not accuracy: an answer that contains the reference value is not proven correct, and one that
does not may still be right in different words. It needs no server changes (unlike evaluate_live_api, which needs the
evaluation API enabled). Never prints or stores the API token.

Run (inside the compose network, API token from the environment):
    API_AUTH_TOKEN=... python -m evaluation.quick_check --questions evaluation/full-schema-300-drugcen-2027-20261003/questions.json \\
        --url http://api:8000 --output benchmarks/quick-check-full-schema
Resumable: rerun the same command and finished questions are skipped. Use --limit 10 first.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import time

ABSTAIN_PREFIX = 'The requested information could not be verified'


def normalize(text):
    return re.sub(r'\s+', ' ', str(text)).casefold().strip()


def classify(answer):
    """answered / abstained / clarification, from the answer text alone."""
    text = (answer or '').strip()
    if not text or text.startswith(ABSTAIN_PREFIX):
        return 'abstained'
    if re.search(r'(?i)could refer to|which drug do you mean|which formulation', text[:300]) and len(text) < 400:
        return 'clarification'
    return 'answered'


def answer_body(answer):
    """The answer text without citation markers and the Sources footer, which can contain digits by accident."""
    text = re.split(r'\n\s*Sources:', answer or '', maxsplit=1)[0]
    return re.sub(r'\[\d+\]', ' ', text)


def stem(word):
    return re.sub(r'(?:ing|es|s)$', '', word) if len(word) > 4 else word


def fact_hit(answer, facts):
    """True when an expected value appears as a whole token or phrase (so 'g' does not match 'evidence').

    A long value (four or more words) also counts when at least 80% of its words, ignoring plural and -ing endings, appear
    in the answer, so a correct paraphrase of a definition is not scored as a miss.
    """
    text = normalize(answer_body(answer))
    words = {stem(w) for w in re.findall(r'[a-z0-9]+', text)}
    for fact in facts or []:
        fact = normalize(fact)
        if not fact:
            continue
        if re.search(r'(?<![a-z0-9])' + re.escape(fact) + r'(?![a-z0-9])', text):
            return True
        tokens = [stem(w) for w in re.findall(r'[a-z0-9]+', fact)]
        if len(tokens) >= 4 and sum(t in words for t in tokens) / len(tokens) >= 0.8:
            return True
    return False


def is_weak(facts):
    """Expected values of one or two characters are likely to appear by chance, so a hit is weak evidence."""
    return not facts or max(len(normalize(f)) for f in facts) <= 2


def check(questions, ask, done=()):
    """Yield one result row per question not already done. `ask(question_text)` returns the answer text."""
    for q in questions:
        if q['id'] in done:
            continue
        started = time.monotonic()
        try:
            answer, error = ask(q['question']), None
        except Exception as exc:  # a failed request is recorded, not hidden, and does not stop the run
            answer, error = '', f'{type(exc).__name__}: {exc}'[:200]
        yield {'id': q['id'], 'source_table': q.get('source_table') or q.get('category'), 'object_type': q.get('source_object_type'),
               'category': q.get('category'), 'question': q['question'], 'answer': answer, 'error': error,
               'status': 'error' if error else classify(answer),
               # an abstention or clarification can never count as having produced the value
               'fact_hit': (not error) and classify(answer) == 'answered' and fact_hit(answer, q.get('reference_facts')),
               'weak_check': is_weak(q.get('reference_facts')),
               'reference_facts': q.get('reference_facts'), 'seconds': round(time.monotonic() - started, 2)}


def summarize(rows):
    def block(subset):
        n = len(subset)
        answered = sum(r['status'] == 'answered' for r in subset)
        return {'questions': n, 'answered': answered, 'abstained': sum(r['status'] == 'abstained' for r in subset),
                'clarification': sum(r['status'] == 'clarification' for r in subset), 'errors': sum(r['status'] == 'error' for r in subset),
                'expected_value_in_answer': sum(r['fact_hit'] for r in subset),
                'expected_value_in_answer_reliable': sum(r['fact_hit'] and not r.get('weak_check') for r in subset),
                'weak_checks': sum(bool(r.get('weak_check')) for r in subset),
                'answered_rate': round(answered / n, 3) if n else None,
                'expected_value_rate': round(sum(r['fact_hit'] for r in subset) / n, 3) if n else None}
    by_table, by_type = defaultdict(list), defaultdict(list)
    for r in rows:
        by_table[r['source_table']].append(r)
        by_type[r['object_type']].append(r)
    seconds = sorted(r['seconds'] for r in rows)
    return {'overall': block(rows), 'by_object_type': {k: block(v) for k, v in sorted(by_type.items(), key=lambda x: str(x[0]))},
            'by_table': {k: block(v) for k, v in sorted(by_table.items(), key=lambda x: str(x[0]))},
            'tables_with_no_expected_value': sorted(k for k, v in by_table.items() if not any(r['fact_hit'] for r in v)),
            'latency_p50_seconds': seconds[len(seconds) // 2] if seconds else None,
            'latency_p95_seconds': seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))] if seconds else None,
            'note': 'Mechanical check only: needs independent review before any accuracy claim.'}


def rescore(rows, questions):
    """Recompute status and hit flags for stored answers with the current matcher. No API calls, no cost."""
    facts = {q['id']: q.get('reference_facts') for q in questions}
    out = []
    for row in rows:
        row = dict(row)
        if not row.get('error'):
            row['status'] = classify(row['answer'])
            row['fact_hit'] = row['status'] == 'answered' and fact_hit(row['answer'], facts.get(row['id']))
        row['weak_check'] = is_weak(facts.get(row['id']))
        out.append(row)
    return out


def make_ask(url, token, timeout=180, retries=3):
    import requests

    def ask(question):
        for attempt in range(retries):
            response = requests.post(url.rstrip('/') + '/api/chat', json={'question': question}, timeout=timeout,
                                     headers={'Authorization': 'Bearer ' + token})
            if response.status_code in (429, 503) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()['answer']
    return ask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rescore', action='store_true', help='re-score stored answers with the current matcher; makes no API calls')
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / 'results.jsonl'
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()] if results.exists() else []
    if args.rescore:
        rows = rescore(rows, questions)
        results.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        summary = summarize(rows)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary['overall'], indent=2))
        print('Tables where no expected value appeared:', ', '.join(summary['tables_with_no_expected_value']) or 'none')
        return
    token = os.environ.get('API_AUTH_TOKEN')
    if not token:
        parser.error('Set API_AUTH_TOKEN in the environment')
    with results.open('a') as handle:
        for row in check(questions, make_ask(args.url, token), {r['id'] for r in rows}):
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            rows.append(row)
            print(f"{len(rows):>3}/{len(questions)} {row['status']:<13} hit={int(row['fact_hit'])} {row['seconds']:>5}s {row['source_table']}", flush=True)
    summary = summarize(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['overall'], indent=2))
    print('Tables where no expected value appeared:', ', '.join(summary['tables_with_no_expected_value']) or 'none')


if __name__ == '__main__':
    main()
