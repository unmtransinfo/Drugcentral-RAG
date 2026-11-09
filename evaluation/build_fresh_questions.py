"""Freeze a seeded, drug-disjoint question set from the configured source database.

Selection uses source availability only, never chatbot answers or benchmark scores.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chatbot.retrieval import EvidenceIndex

# Question templates per category.
HEADERS = {
 'targets': ('Biological Targets', 'Name one biological target recorded for {d}; include its gene symbol if recorded.'),
 'indications': ('Indications', 'Name one indication recorded for {d} in this DrugCentral snapshot.'),
 'contraindications': ('Contraindications', 'Name one contraindication listed for {d} in this DrugCentral snapshot.'),
 'approvals': ('Regulatory Approvals', 'Give one regulatory agency and its recorded approval year for {d}.'),
 'adme': ('ADME / Pharmacokinetics Properties', 'Give one recorded ADME property of {d}, including its value and units if present.'),
 'pharmacology': ('Pharmacologic Class (EPC)', 'Name one established pharmacologic class recorded for {d}.'),
 'atc': ('ATC Classifications', 'Give one ATC code and its associated description recorded for {d}.'),
 'structures': ('Protein Data Bank (PDB) Co-crystallized Codes', 'Give one PDB co-crystallized structure code recorded for {d}.'),
 'interactions': ('Drug-Drug Interactions', 'What interaction severity is recorded between {d} and {entity}?'),
 'adverse_events': ('Clinical Adverse Events (FAERS Statistics)', 'For {d}, what LLR and case count are recorded for the FAERS event {entity}? Describe this as a report association, not proof of causation.'),
}


def build_questions(index, previous, seed=20260928, per_category=24):
    rng = random.Random(seed)
    excluded = {drug for q in previous for drug in ([q['drug']] if q.get('drug') else q.get('drugs', []))}
    # Also exclude resolved drug partners and aliases from the earlier questions.
    for q in previous:
        for name in index.mentions(q['question']):
            excluded.update(index.candidates(name))
    old_text = {' '.join(q['question'].casefold().split()) for q in previous}
    pools = {}
    for category, (header, _) in HEADERS.items():
        candidates = []
        for drug in sorted(index.sections):
            if drug in excluded or drug not in index.ids:
                continue
            facts = list(dict.fromkeys(fact for label, fact in index.sections[drug].get(category, []) if label == header))
            if category == 'interactions':
                facts = [f for f in facts if ' [' in f and ']' in f
                         and not (set(index.candidates(f.split(' [', 1)[0])) & excluded)]
            elif category == 'adverse_events':
                facts = [f for f in facts if ' (LLR:' in f and 'cases)' in f]
            if facts and len('\n'.join(facts)) <= 10000:
                candidates.append((drug, facts))
        rng.shuffle(candidates)
        pools[category] = candidates
    # Allocate scarce source categories first so all primary drugs are unique.
    selected = set()
    questions = []
    for category in sorted(HEADERS, key=lambda c: (len(pools[c]), c)):
        header, template = HEADERS[category]
        count = 0
        for drug, all_facts in pools[category]:
            if drug in selected:
                continue
            entity = ''
            facts = all_facts
            if category in ('interactions', 'adverse_events'):
                separator = ' [' if category == 'interactions' else ' (LLR:'
                entity = rng.choice(all_facts).split(separator, 1)[0]
                facts = [f for f in all_facts if f.split(separator, 1)[0] == entity]
            question = template.format(d=drug, entity=entity)
            if ' '.join(question.casefold().split()) in old_text:
                continue
            count += 1
            selected.add(drug)
            questions.append(dict(id=f'fresh-{seed}-{category}-{count:03d}', category=category,
                question=question, drug=drug, section_header=header,
                reference_facts=facts, reference_answer=facts[0],
                answer_rule='One matching fact suffices; all additional factual claims must agree with reference evidence. Preserve requested values and qualifiers.',
                source_url=f'https://drugcentral.org/drugcard/{index.ids[drug]}',
                dataset_version=','.join(sorted(index.versions)),
                reference_sha256=hashlib.sha256(json.dumps(facts,ensure_ascii=False).encode()).hexdigest(),
                review={'status':'pending','reviewer':None}))
            if count == per_category:
                break
        if count != per_category:
            raise ValueError(f'Only {count} eligible unused drugs for {category}; need {per_category}')
    questions.sort(key=lambda q: (list(HEADERS).index(q['category']), q['id']))
    assert len(questions) == len(HEADERS)*per_category
    assert len({q['drug'] for q in questions}) == len(questions)
    assert not ({q['drug'] for q in questions} & excluded)
    return questions, excluded, {c:len(v) for c,v in pools.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exclude', type=Path, default=ROOT/'benchmarks/web-judge-240/human_review.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--per-category', type=int, default=24)
    p.add_argument('--also-exclude', type=Path, action='append', default=[])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory; existing sets are never overwritten')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from chatbot.storage import load_documents, backend_name
    from chatbot.embedding_config import embedding_model_id
    if backend_name() != 'pgvector' or not os.getenv('STRUCTURED_COLLECTION'):
        raise ValueError('Requires the configured structured pgvector source database')
    print('Reading configured DrugCentral source database...',flush=True)
    data = load_documents(None, embedding_model_id())
    index = EvidenceIndex(data)
    del data
    if args.per_category < 1: raise ValueError('--per-category must be positive')
    exclusion_files = [args.exclude, *args.also_exclude]
    previous = []
    for path in exclusion_files:
        value = json.loads(path.read_text())
        previous.extend(value['questions'] if isinstance(value, dict) else value)
    questions, excluded, pools = build_questions(index, previous, args.seed, args.per_category)
    manifest = dict(questions=len(questions), unique_primary_drugs=len({q['drug'] for q in questions}),
        categories=dict(Counter(q['category'] for q in questions)), seed=args.seed,
        source_drugs=len(index.sections), dataset_versions=sorted(index.versions),
        excluded_drugs=sorted(excluded), eligible_drugs_by_category=pools,
        excluded_file_sha256=hashlib.sha256(args.exclude.read_bytes()).hexdigest(),
        exclusion_files_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in exclusion_files},
        scope='Fresh primary drugs and source facts, with the same ten question templates as the previous benchmark. Source-derived factual lookup set pending domain review; not a new paraphrase/clinical-reasoning test. No answer generation or score-based selection.',
        selection=f'{args.per_category} per category; {len(questions)} distinct primary drugs; exclude earlier primary drugs and resolved drug mentions; seed-shuffled candidates; random interaction partners/FAERS events; nonempty source sections with <=10000 reference characters.')
    args.output.mkdir(parents=True)
    encoded=json.dumps(questions,indent=2,ensure_ascii=False)+'\n'
    (args.output/'questions.json').write_text(encoded)
    manifest['questions_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='excluded_drugs'},indent=2))


if __name__ == '__main__':
    main()
