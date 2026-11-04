"""Compare a new quick_check run with an earlier answer set on the SAME questions, to find regressions.

Both sets of answers are scored with the same matcher (evaluation.quick_check), so a difference reflects the answers, not the
scoring. Prints, per category, how many questions each run answered and matched, then lists the questions that got worse
(the old answer contained the expected value and the new one does not) and the ones that improved.

Run (the new results come from quick_check --output DIR; the old answers from baseline-v4-answers.json):
    python -m evaluation.compare_runs --questions evaluation/fresh-240-20260928/questions.json \\
        --old evaluation/fresh-240-20260928/baseline-v4-answers.json --new benchmarks/quick-check-fresh-240/results.jsonl
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import re

from evaluation.quick_check import answer_body, classify, fact_hit


def load_old(path):
    data = json.loads(Path(path).read_text())
    rows = data['rows'] if isinstance(data, dict) else data
    return {r['id']: r['answer'] for r in rows}


def load_new(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r['id']: r['answer'] for r in rows if not r.get('error')}


def words(answer):
    """Content words of an answer without citation markers, sources and very short words."""
    return set(re.findall(r'[a-z0-9]{3,}', answer_body(answer).casefold()))


def similarity(a, b):
    """Jaccard overlap of content words: 1.0 for the same content, near 0 for unrelated answers."""
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def score(answer, facts):
    status = classify(answer)
    return {'status': status, 'hit': status == 'answered' and fact_hit(answer, facts)}


def compare(questions, old, new):
    by_category = defaultdict(lambda: defaultdict(int))
    worse, better, missing, changed = [], [], [], []
    for q in questions:
        if q['id'] not in old or q['id'] not in new:
            missing.append(q['id'])
            continue
        o, n = score(old[q['id']], q['reference_facts']), score(new[q['id']], q['reference_facts'])
        sim = similarity(old[q['id']], new[q['id']])
        if sim < 0.6 or o['status'] != n['status']:
            changed.append((sim, q, old[q['id']], new[q['id']], o['status'], n['status']))
        c = by_category[q['category']]
        c['questions'] += 1
        c['old_hit'] += o['hit']
        c['new_hit'] += n['hit']
        c['old_abstained'] += o['status'] == 'abstained'
        c['new_abstained'] += n['status'] == 'abstained'
        if o['hit'] and not n['hit']:
            worse.append((q, new[q['id']], n['status']))
            c['worse'] += 1
        elif n['hit'] and not o['hit']:
            better.append((q, old[q['id']]))
            c['better'] += 1
    return by_category, worse, better, missing, sorted(changed, key=lambda c: c[0])


def totals(by_category):
    total = defaultdict(int)
    for c in by_category.values():
        for k, v in c.items():
            total[k] += v
    return total


def head(text, n=170):
    return ' '.join(str(text).split())[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--show', type=int, default=15)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    by_category, worse, better, missing, changed = compare(questions, load_old(args.old), load_new(args.new))
    total = totals(by_category)
    print(f"{'category':16}{'Qs':>4}{'old hit':>9}{'new hit':>9}{'worse':>7}{'better':>8}{'old abst':>10}{'new abst':>10}")
    for name, c in sorted(by_category.items()):
        print(f"{name:16}{c['questions']:>4}{c['old_hit']:>9}{c['new_hit']:>9}{c['worse']:>7}{c['better']:>8}{c['old_abstained']:>10}{c['new_abstained']:>10}")
    print(f"{'ALL':16}{total['questions']:>4}{total['old_hit']:>9}{total['new_hit']:>9}{total['worse']:>7}{total['better']:>8}"
          f"{total['old_abstained']:>10}{total['new_abstained']:>10}")
    if missing:
        print(f'\n{len(missing)} questions missing from one of the runs (first: {missing[:3]})')
    print(f"\nWORSE ({len(worse)}): the old answer had the expected value, the new one does not")
    for q, answer, status in worse[:args.show]:
        print(f"  [{q['category']}] {head(q['question'], 110)}\n     expected: {head(q['reference_facts'], 80)}\n     new ({status}): {head(answer)}")
    print(f"\nCHANGED ANSWERS ({len(changed)}): content overlap below 60% or a different status, least similar first.")
    print("The matcher misses paraphrased values in some categories, so read these to find real regressions.")
    for sim, q, old_answer, new_answer, old_status, new_status in changed[:args.show]:
        print(f"  [{q['category']}] overlap {sim:.0%} ({old_status} -> {new_status}) {head(q['question'], 100)}\n"
              f"     expected: {head(q['reference_facts'], 80)}\n     old: {head(old_answer)}\n     new: {head(new_answer)}")
    print(f"\nBETTER ({len(better)}) (showing {min(len(better), 5)})")
    for q, old_answer in better[:5]:
        print(f"  [{q['category']}] {head(q['question'], 110)}\n     expected: {head(q['reference_facts'], 80)}\n     old: {head(old_answer)}")


if __name__ == '__main__':
    main()
