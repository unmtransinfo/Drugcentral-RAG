"""Build and validate the reviewed evaluation set against the current snapshot.

Expectations are derived mechanically from the versioned evidence snapshot and
stay review.status=pending until a domain reviewer signs off. Rebuilding merges
existing non-pending review fields by question id, so reviewer work survives
regeneration. Abstention is expected wherever the snapshot lacks the requested
evidence; missing source facts are never counted as answerable.

Run: python evaluation_set.py [--force]
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from dotenv import load_dotenv
from chatbot.retrieval import EvidenceIndex, source_grounded_answer
from chatbot.question_retrieval import filter_records

SET_PATH = Path('evaluation/questions.json')

DEV = ['metformin', 'atorvastatin', 'warfarin', 'omeprazole']
HELDOUT = ['amlodipine', 'aspirin', 'ibuprofen', 'lisinopril', 'losartan', 'simvastatin',
           'carboplatin', 'fluoxetine', 'digoxin', 'amoxicillin', 'levothyroxine', 'clopidogrel']
UNSEEN = ['metoprolol', 'gabapentin', 'prednisone', 'tramadol', 'sertraline', 'pantoprazole',
          'furosemide', 'hydrochlorothiazide']
BRANDS = {'Glucophage': 'metformin', 'Tylenol': 'acetaminophen', 'Advil': 'ibuprofen',
          'Lipitor': 'atorvastatin', 'Coumadin': 'warfarin', 'Prilosec': 'omeprazole',
          'Norvasc': 'amlodipine', 'Synthroid': 'levothyroxine', 'Plavix': 'clopidogrel',
          'Prozac': 'fluoxetine', 'Zocor': 'simvastatin', 'Cozaar': 'losartan'}
UNKNOWN = ['Obscuritol', 'Fakemed', 'Notarealstat', 'Zorblax', 'Placebomin']
PAIRS = [('metformin', 'warfarin'), ('atorvastatin', 'clopidogrel'), ('ibuprofen', 'lisinopril'),
         ('omeprazole', 'fluoxetine'), ('digoxin', 'amoxicillin'), ('simvastatin', 'amlodipine')]
SECTION_TEMPLATES = [
    ('targets', 'What targets are associated with {d}?', 'targets', None),
    ('indications', 'What is {d} used for?', 'indications', None),
    ('off_label', 'List off-label uses of {d}.', 'indications', 'Off-label Uses:'),
    ('contraindications', 'What are the contraindications of {d}?', 'contraindications', None),
    ('interactions', 'What drug interactions are documented for {d}?', 'interactions', None),
    ('adverse_events', 'What side effects are reported for {d}?', 'adverse_events', None),
    ('approvals', 'What regulatory approvals are recorded for {d}?', 'approvals', None),
    ('adme', 'What is the half-life of {d}?', 'adme', 'half-life'),
]
CORE = {'targets', 'indications', 'interactions'}
FILTERS = [
    ([{'section': 'targets', 'term': 'DPP4'}], 'How many drugs have DPP4 as a target?'),
    ([{'section': 'indications', 'term': 'hypertension'}], 'How many drugs are indicated for hypertension?'),
    ([{'section': 'targets', 'term': 'cyclooxygenase'}, {'section': 'indications', 'term': 'pain'}],
     'How many drugs target cyclooxygenase and are indicated for pain?'),
    ([{'section': 'indications', 'term': 'type 2 diabetes'}], 'How many drugs treat type 2 diabetes?'),
    ([{'section': 'contraindications', 'term': 'pregnancy'}], 'How many drugs list pregnancy as a contraindication?'),
    ([{'section': 'targets', 'term': 'xyzzy-not-a-target'}], 'How many drugs target xyzzy-not-a-target?'),
]
QUALIFIED = [('What is metformin used for in children?', ['metformin']),
             ('Is warfarin safe during pregnancy?', ['warfarin']),
             ('What are the contraindications of ibuprofen in elderly patients?', ['ibuprofen']),
             ('Can I take omeprazole and clopidogrel together?', ['omeprazole', 'clopidogrel']),
             ('What dose of digoxin is safe?', ['digoxin']),
             ('List off-label uses of gabapentin for anxiety.', ['gabapentin'])]


def resolve(index, name):
    candidates = index.candidates(name)
    return candidates[0] if len(candidates) == 1 else None


def cite(index, drug):
    return f'https://drugcentral.org/drugcard/{index.ids[drug]}' if drug in index.ids else None




def build(index):
    entries, skipped = [], []

    def section_entry(category, question, asked, section, needle):
        drug = resolve(index, asked)
        if not drug:
            skipped.append({'asked': asked, 'reason': 'unresolved or ambiguous'})
            return
        evidence = index.evidence(drug, section)
        text, grounded = evidence['text'], source_grounded_answer(index, question) is not None
        expect = {'cite': [cite(index, drug)] if cite(index, drug) else [],
                  'route': 'grounded' if grounded else 'agent',
                  'sections': [section], 'needle': needle}
        if text and (needle is None or needle.lower() in text.lower()):
            expect['behavior'] = 'answer'
            if grounded:
                first_fact = next(l.strip() for l in text.splitlines() if l.strip().startswith('- '))
                expect['verbatim'] = True
                expect['must_contain'] = [first_fact[2:].strip()]
        else:
            expect['behavior'] = 'abstain'
            expect['abstain_reason'] = f'no source evidence in section {section}'
        entries.append(make(len(entries) + 1, category, question, [drug], expect))

    for drug in DEV + HELDOUT:
        for category, template, section, needle in SECTION_TEMPLATES:
            section_entry(category, template.format(d=drug), drug, section, needle)
    for drug in UNSEEN:
        for category, template, section, needle in SECTION_TEMPLATES:
            if category in CORE:
                section_entry(category, template.format(d=drug), drug, section, needle)
    for brand, official in BRANDS.items():
        section_entry('synonym', f'What is {brand} used for?', brand, 'indications', None)
    for name in sorted((n for n, c in index.names.items()
                        if len(c) > 1 and n.isalpha() and 5 <= len(n) <= 12),
                       key=lambda n: (-len(index.names[n]), n))[:6]:
        entries.append(make(len(entries) + 1, 'ambiguous', f'What is {name} used for?',
                            sorted(index.names[name]),
                            {'behavior': 'clarify', 'route': 'grounded',
                             'candidates': sorted(index.names[name])},
                            note='name maps to multiple drugs; the app must ask which one'))
    for first, second in PAIRS:
        urls = [cite(index, d) for d in (resolve(index, first), resolve(index, second)) if d]
        if len(urls) == 2:
            entries.append(make(len(entries) + 1, 'multiple_drugs',
                f'What targets are associated with {first} and {second}?', [first, second],
                {'behavior': 'answer', 'cite': urls, 'route': 'agent'},
                note='comparison; both drugs must be cited, neither invented'))
    for name in UNKNOWN:
        entries.append(make(len(entries) + 1, 'missing_information', f'What is {name} used for?',
                            [], {'behavior': 'abstain', 'route': 'grounded',
                                 'abstain_reason': 'unknown drug name'}))
    for filters, question in FILTERS:
        total = filter_records(index, filters, limit=1)['total_matches']
        entries.append(make(len(entries) + 1, 'filtered_count', question, [],
                            {'behavior': 'answer', 'must_contain': [str(total)],
                             'expected_count': total, 'route': 'agent'},
                            note='literal token matching scope must be disclosed'))
    for question, drugs in QUALIFIED:
        urls = [cite(index, d) for d in (resolve(index, x) for x in drugs) if d]
        entries.append(make(len(entries) + 1, 'qualified', question, drugs,
                            {'behavior': 'answer', 'cite': urls, 'route': 'agent'},
                            note='qualified question; verify no overgeneralization beyond evidence'))
    return entries, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Discard existing review fields')
    args = parser.parse_args()
    load_dotenv()
    from chatbot.embedding_config import create_embeddings, embedding_model_id
    from chatbot.storage import load_documents
    index = EvidenceIndex(load_documents(create_embeddings(), embedding_model_id()))
    entries, skipped = build(index)
    if SET_PATH.exists() and not args.force:
        old = {e['id']: e for e in json.loads(SET_PATH.read_text())['questions']}
        for entry in entries:
            if entry['id'] in old and old[entry['id']]['review']['status'] != 'pending':
                entry['review'] = old[entry['id']]['review']
    counts = Counter(e['category'] for e in entries)
    if not 100 <= len(entries) <= 200:
        raise ValueError(f'Expected 100-200 questions, built {len(entries)}')
    required = {'targets', 'indications', 'interactions', 'synonym', 'ambiguous',
                'multiple_drugs', 'missing_information', 'filtered_count', 'qualified'}
    if not required <= set(counts):
        raise ValueError(f'Missing categories: {required - set(counts)}')
    if not any(e['expect']['behavior'] == 'abstain' and e['drugs'] for e in entries):
        raise ValueError('No missing-section abstention cases were derived')
    SET_PATH.parent.mkdir(exist_ok=True)
    payload = {'version': 1, 'dataset_versions': sorted(index.versions),
               'scope': 'Mechanically derived expectations; pending domain review.',
               'categories': dict(counts), 'skipped': skipped, 'questions': entries}
    SET_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps({'questions': len(entries), 'categories': dict(counts),
                      'skipped': len(skipped)}, indent=2))


def make(seq, category, question, drugs, expect, note=None):
    return {'id': f'{category}-{seq:03d}', 'category': category, 'question': question,
            'drugs': drugs, 'expect': expect,
            'review': {'status': 'pending', 'reviewer': None, 'decision': None, 'notes': note}}


if __name__ == '__main__':
    main()
