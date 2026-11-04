"""Build the 110-question complex benchmark (multi-part, multi-drug, multi-hop, partial, unanswerable).

Questions are sampled deterministically from the complete DrugCentral source database and the
gold facts are read straight from source tables, independent of the vector collection and the
graph. Output is UNREVIEWED: have an independent reviewer sign off before treating it as gold.

Run: python -m evaluation.build_complex_questions --output evaluation/complex-110
Env: SOURCE_DATABASE_URL (or SOURCE_PG*), optional LABEL_DATABASE.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3

from chatbot.question_retrieval import requested_sections

COMPOSITION = {'independent_2part': 20, 'independent_3part': 15, 'multi_drug_comparison': 15, 'cross_section': 15,
               'true_multi_hop': 15, 'long_label': 10, 'partially_answerable': 10, 'completely_unanswerable': 10}
# Noun phrases that must trigger exactly their own field in the planner (checked below).
NOUN = {'targets': 'targets', 'indications': 'indications', 'contraindications': 'contraindications',
        'atc': 'ATC classification', 'approvals': 'regulatory approvals', 'properties': 'molecular formula',
        'synonyms': 'synonyms', 'structures': 'PDB structure codes'}
FACT_SQL = {
    'targets': "SELECT a.struct_id, t.name FROM act_table_full a JOIN target_dictionary t ON t.id = a.target_id",
    'indications': "SELECT struct_id, concept_name FROM omop_relationship WHERE relationship_name = 'indication'",
    'contraindications': "SELECT struct_id, concept_name FROM omop_relationship WHERE relationship_name = 'contraindication'",
    'atc': "SELECT struct_id, trim(atc_code) FROM struct2atc",
    'approvals': "SELECT struct_id, type FROM approval",
    'properties': "SELECT id, cd_formula FROM structures WHERE cd_formula IS NOT NULL",
    'synonyms': "SELECT id, name FROM synonyms",
    'structures': "SELECT struct_id, pdb FROM pdb",
}
LABEL_SECTIONS = {'adverse_events': 'adverse_reactions', 'warnings': 'warnings', 'dosage': 'dosage'}
FAKE_DRUGS = ['zorblaxinol', 'quenvastatin', 'trimexolacet', 'veltranimab', 'orpheniclor']
UNSUPPORTED_FIELDS = ['retail price', 'manufacturing plant location', 'storage temperature', 'shelf life',
                      'customer support phone number']


def check_nouns():
    for section, noun in NOUN.items():
        found = requested_sections('What are the ' + noun + ' of metformin?')
        if found != [section]:
            raise ValueError(f'Noun phrase for {section} routes to {found}')


def load_source(conn):
    """Per-drug facts by section, straight from the source tables."""
    facts = defaultdict(lambda: defaultdict(list))
    with conn.cursor() as cur:
        cur.execute('SELECT id, name FROM structures')
        names = {i: n for i, n in cur.fetchall()}
        for section, sql in FACT_SQL.items():
            cur.execute(sql)
            for struct_id, value in cur.fetchall():
                if value and str(value).strip() and value not in facts[struct_id][section]:
                    facts[struct_id][section].append(str(value).strip())
    return names, facts


def label_drugs(path):
    if not path or not Path(path).exists():
        return {}
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as c:
        rows = c.execute('''SELECT i.struct_id, s.section FROM ingredients i JOIN products p ON p.ndc = i.ndc
            JOIN links l ON l.ndc = p.ndc JOIN sections s ON s.label_id = l.label_id
            WHERE p.ingredient_count = 1 GROUP BY i.struct_id, s.section''').fetchall()
    out = defaultdict(set)
    for struct_id, section in rows:
        out[struct_id].add(section)
    return out


def usable(name):
    return bool(re.fullmatch(r'[a-z][a-z \-]{3,30}', name or ''))


def component(drug, section, facts, label_sections, cid):
    if section in LABEL_SECTIONS:
        answerable = LABEL_SECTIONS[section] in label_sections
        reference = []
    else:
        reference = facts.get(section, [])[:5]
        answerable = bool(reference)
    return {'id': cid, 'drug': drug, 'section': section, 'answerable': answerable, 'reference_facts': reference}


def join_nouns(nouns):
    return nouns[0] if len(nouns) == 1 else ', '.join(nouns[:-1]) + ' and ' + nouns[-1]


def overall(components):
    answerable = sum(c['answerable'] for c in components)
    return 'full' if answerable == len(components) else 'partial' if answerable else 'none'


def build(names, facts, labels, seed=20261004):
    check_nouns()
    rng = random.Random(seed)
    rich = sorted(i for i in names if usable(names[i]) and len(facts[i]) >= 4 and 'targets' in facts[i] and 'indications' in facts[i])
    if len(rich) < 80:
        raise ValueError(f'Only {len(rich)} drugs have enough source coverage to sample')
    pool = list(rich); rng.shuffle(pool)
    used = set()
    def take(predicate=lambda i: True):
        for i in pool:
            if i not in used and predicate(i):
                used.add(i); return i
        raise ValueError('Not enough eligible drugs for the requested composition')
    questions = []
    def add(category, question, components, **extra):
        questions.append({'id': f'cx{len(questions) + 1:03d}', 'category': category, 'question': question,
                          'answerable': overall(components), 'components': components, **extra})
    structured = ['targets', 'indications', 'atc', 'approvals', 'contraindications', 'properties', 'synonyms']
    for category, count in (('independent_2part', COMPOSITION['independent_2part']), ('independent_3part', COMPOSITION['independent_3part'])):
        size = 2 if category.endswith('2part') else 3
        for _ in range(count):
            i = take(lambda i: sum(s in facts[i] for s in structured) >= size)
            sections = rng.sample([s for s in structured if s in facts[i]], size)
            add(category, f'What are the {join_nouns([NOUN[s] for s in sections])} of {names[i]}?',
                [component(names[i], s, facts[i], labels.get(i, set()), f'c{k}') for k, s in enumerate(sections, 1)], drugs=[names[i]])
    for _ in range(COMPOSITION['multi_drug_comparison']):
        a = take(lambda i: 'targets' in facts[i]); b = take(lambda i: 'targets' in facts[i])
        sections = rng.sample(['targets', 'indications', 'atc'], 2)
        components = [component(names[d], s, facts[d], labels.get(d, set()), f'c{k}')
                      for k, (d, s) in enumerate(((d, s) for d in (a, b) for s in sections), 1)]
        add('multi_drug_comparison', f'Compare the {join_nouns([NOUN[s] for s in sections])} of {names[a]} and {names[b]}.',
            components, drugs=[names[a], names[b]])
    with_labels = lambda i: 'adverse_reactions' in labels.get(i, set())
    for _ in range(COMPOSITION['cross_section']):
        i = take(with_labels)
        structured_section = rng.choice([s for s in ('indications', 'targets', 'atc', 'approvals') if s in facts[i]])
        add('cross_section', f'What are the {NOUN[structured_section]} and common side effects of {names[i]}?',
            [component(names[i], structured_section, facts[i], labels.get(i, set()), 'c1'),
             component(names[i], 'adverse_events', facts[i], labels.get(i, set()), 'c2')], drugs=[names[i]])
    for _ in range(COMPOSITION['long_label']):
        i = take(lambda i: {'adverse_reactions', 'warnings', 'dosage'} <= labels.get(i, set()))
        add('long_label', f'What are the common adverse reactions, warnings and dosage information in the product label of {names[i]}?',
            [component(names[i], s, facts[i], labels.get(i, set()), f'c{k}') for k, s in enumerate(('adverse_events', 'warnings', 'dosage'), 1)],
            drugs=[names[i]])
    for _ in range(COMPOSITION['partially_answerable']):
        missing = rng.choice(['structures', 'contraindications'])
        i = take(lambda i: 'targets' in facts[i] and missing not in facts[i])
        add('partially_answerable', f'What are the targets and {NOUN[missing]} of {names[i]}?',
            [component(names[i], 'targets', facts[i], set(), 'c1'), component(names[i], missing, facts[i], set(), 'c2')], drugs=[names[i]])
    for k in range(COMPOSITION['completely_unanswerable']):
        if k % 2 == 0:
            i = take()
            field = UNSUPPORTED_FIELDS[(k // 2) % len(UNSUPPORTED_FIELDS)]
            add('completely_unanswerable', f'What is the {field} of {names[i]}?',
                [{'id': 'c1', 'drug': names[i], 'section': 'unsupported', 'answerable': False, 'reference_facts': []}], drugs=[names[i]])
        else:
            fake = FAKE_DRUGS[k // 2]
            add('completely_unanswerable', f'What are the targets and indications of {fake}?',
                [{'id': 'c1', 'drug': fake, 'section': 'targets', 'answerable': False, 'reference_facts': []},
                 {'id': 'c2', 'drug': fake, 'section': 'indications', 'answerable': False, 'reference_facts': []}], drugs=[fake])
    return questions


def multi_hop(conn, names, rng, count=15):
    """Graph-style questions with gold SETS computed by direct SQL (never from the graph)."""
    out = []
    with conn.cursor() as cur:
        cur.execute('SELECT a.struct_id, a.target_id FROM act_table_full a')
        drug_targets = defaultdict(set); target_drugs = defaultdict(set)
        for s, t in cur.fetchall():
            drug_targets[s].add(t); target_drugs[t].add(s)
        cur.execute("SELECT struct_id, trim(atc_code) FROM struct2atc")
        drug_atc = defaultdict(set); group_drugs = defaultdict(set)
        for s, code in cur.fetchall():
            drug_atc[s].add(code); group_drugs[code[:5]].add(s)
        # Same semantics as the graph template: any listed symbol, any organism.
        cur.execute("SELECT upper(trim(g)), a.struct_id FROM act_table_full a, unnest(string_to_array(a.gene, '|')) g WHERE a.gene IS NOT NULL")
        gene_drugs = defaultdict(set)
        for gene, s in cur.fetchall():
            if gene:
                gene_drugs[gene].add(s)
    def shared(s):
        return {o for t in drug_targets[s] for o in target_drugs[t]} - {s}
    def same_atc(s):
        return {o for c in drug_atc[s] for o in group_drugs[c[:5]]} - {s}
    candidates = sorted(i for i in names if usable(names[i]))
    rng.shuffle(candidates)
    plan = [('shared_target_drugs', 5), ('same_atc_drugs', 5), ('atc_for_target', 5)]
    for intent, n in plan:
        made = 0
        if intent == 'atc_for_target':
            genes = sorted(g for g, d in gene_drugs.items() if re.fullmatch(r'[A-Z][A-Z0-9]{2,8}', g) and 3 <= len(d) <= 40)
            rng.shuffle(genes)
            for gene in genes:
                groups = {c[:5] for s in gene_drugs[gene] for c in drug_atc[s]}
                if 2 <= len(groups) <= 25 and made < n:
                    out.append({'category': 'true_multi_hop', 'intent': intent, 'question': f'What ATC classes do drugs targeting {gene} have?',
                                'drugs': [], 'reference_set': sorted(groups)}); made += 1
            continue
        for i in candidates:
            related = shared(i) if intent == 'shared_target_drugs' else same_atc(i)
            if 2 <= len(related) <= 30 and made < n:
                text = 'Which drugs share targets with' if intent == 'shared_target_drugs' else 'Which drugs are in the same ATC class as'
                out.append({'category': 'true_multi_hop', 'intent': intent, 'question': f'{text} {names[i]}?', 'drugs': [names[i]],
                            'reference_set': sorted(names[o] for o in related)}); made += 1
        if made < n:
            raise ValueError(f'Only {made} {intent} questions available')
    return out[:count]


def assemble(conn, labels_path, seed=20261004):
    names, facts = load_source(conn)
    labels = label_drugs(labels_path)
    questions = build(names, facts, labels, seed)
    questions = [q for q in questions if q['category'] != 'true_multi_hop']
    hops = multi_hop(conn, names, random.Random(seed + 1))
    for hop in hops:
        components = [{'id': 'c1', 'drug': (hop['drugs'] or [None])[0], 'section': 'relationship', 'answerable': True,
                       'reference_facts': hop['reference_set'][:50]}]
        questions.append({'category': 'true_multi_hop', 'question': hop['question'], 'answerable': 'full',
                          'components': components, 'drugs': hop['drugs'], 'graph_intent': hop['intent'],
                          'reference_set': hop['reference_set']})
    order = list(COMPOSITION)
    questions.sort(key=lambda q: (order.index(q['category']), q['question']))
    for n, q in enumerate(questions, 1):
        q['id'] = f'cx{n:03d}'
    return questions


def validate(questions):
    counts = Counter(q['category'] for q in questions)
    if dict(counts) != COMPOSITION:
        raise ValueError(f'Composition mismatch: {dict(counts)}')
    if len({q['question'] for q in questions}) != len(questions):
        raise ValueError('Duplicate questions')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('evaluation/complex-110'))
    parser.add_argument('--labels', default='data/labels.sqlite')
    parser.add_argument('--seed', type=int, default=20261004)
    args = parser.parse_args()
    from chatbot.database_config import connect_database
    conn = connect_database(source=True)
    try:
        questions = assemble(conn, args.labels, args.seed)
        with conn.cursor() as cur:
            cur.execute('SELECT version, dtime::text FROM dbversion ORDER BY dtime DESC LIMIT 1')
            version, loaded = cur.fetchone()
    finally:
        conn.close()
    counts = validate(questions)
    args.output.mkdir(parents=True, exist_ok=False)
    body = json.dumps(questions, indent=2, ensure_ascii=False)
    (args.output / 'questions.json').write_text(body)
    manifest = {'seed': args.seed, 'dataset_version': f'drugcentral-{version}-{loaded[:10]}', 'composition': counts,
                'total': len(questions), 'questions_sha256': hashlib.sha256(body.encode()).hexdigest(),
                'status': 'UNREVIEWED: gold facts come from source tables; an independent reviewer must approve before use',
                'gold': 'component reference_facts are exact source values; multi-hop reference_set is computed by direct SQL'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=dict))
    print(json.dumps(manifest, indent=2, default=dict))


if __name__ == '__main__':
    main()
