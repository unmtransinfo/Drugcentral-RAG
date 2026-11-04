"""Question-aware retrieval from the loaded, versioned DrugCentral snapshot."""
import re
from chatbot.retrieval import tokens

SECTION_CUES = {
    'properties': r'\b(molecular (?:formula|weight|mass)|chemical formula|CAS|logP|TPSA|Lipinski)\b',
    # "target class / keyword / profile ..." names an entity attribute, not the drug-target relationship.
    'targets': r'\b(targets?(?!\s+(?:class(?:es)?|keywords?|profiles?|components?|development\s+level|accessions?|annotations?))|molecular proteins?|receptors?)\b',
    'indications': r'\b(indications?|used for|treats?|treatments? (?:of|for)|(?:drugs?|medications?|medicines?|therapies) for|off.label)\b',
    'contraindications': r'\b(contraindications?|contraindicated|should not take)\b',
    'adme': r'\b(adme|half.life|fall by half|absorption|bioavailability|clearance|pharmacokinetics|excretion|pka)\b',
    'approvals': r'\b(approvals?|approved|regulatory)\b',
    'interactions': r'\b(interactions?|interact with|taken together)\b',
    'adverse_events': r'\b(side effects?|adverse events?|adverse reactions?|faers)\b',
    'pharmacology': r'\b(mechanism|how .+ works?|pharmacolog\w*)\b',
    'synonyms': r'\b(synonyms?|brand names?|also known)\b',
    'atc': r'\batc\b',
    'structures': r'\b(pdb|protein data bank|co.crystallized|structure codes?)\b',
    'patents': r'\b(patents?|exclusivit\w*)\b',
    # Full-database coverage sections (ingestion/full_coverage.py).
    'ddd': r'\b(ddd|defined daily doses?)\b',
    'identifiers': r'\b(identifiers?|unii|pubchem|chembl|drugbank|inchi(?:key)?|rxcui|mesh id|kegg|chebi|nddf|mmsl|umlscui|snomed ?ct id)\b',
    'structure_info': r'\b(parent (?:compound|molecule)|structure types?)\b',
    'pk_measurements': r'\b(astrazeneca|az pk)\b',
    'veterinary': r'\b(veterinary|animal drugs?|canine|feline|equine|bovine|vet products?)\b',
    'adverse_events_male': r'\b(?:male|men)\b[- ]*(?:patients?|subgroup)',
    'adverse_events_female': r'\b(?:female|women)\b[- ]*(?:patients?|subgroup)',
    'adverse_events_pediatric': r'\b(?:pediatric|paediatric|children|child)\b[- ]*(?:patients?|subgroup)',
    'adverse_events_geriatric': r'\b(?:geriatric|elderly)\b[- ]*(?:patients?|subgroup)',
    'adverse_events_top': r'\b(?:top|strongest|highest)[- ]?(?:ranked)?\b[^?.]{0,30}\bsignals?\b',
    'patent_details': r'\b(orange book|trade names?|applicants?|exclusivity descriptions?)\b',
    'doid_mapping': r'\b(snomed|umls|cui|(?:doid|disease ontology)\b[^?.]{0,20}\bmappings?)\b',
    'lincs': r'\b(lincs|signature similarity)\b',
    'target_profile': r'\b(target class|protein class|protein type|target components?|uniprot|swiss-?prot|accession|target development level|tdl|go terms?|gene ontology|keywords?)\b',
    'ontology': r'\b(go[: ]?\d{3,}|gene ontology|go terms?|keywords?)\b',
    'disease_profile': r'\b(disease profile|disease definition|doid(?![^?.]{0,20}\bmappings?)|disease ontology(?![^?.]{0,20}\bmappings?))\b',
    'references': r'\b(pmid|doi|isbn|journal|publications?|citations?|authors?)\b',
    'dictionary': r'\b(dictionary|lookup table|data sources?|id types?|ref(?:erence)? types?|approval types?|use codes?|exclusivity codes?|(?:definition|meaning|description) of (?:the )?(?:action|approval|id|ref(?:erence)?|protein) types?)\b',
}


# Questions often name a source table directly ("in table dbversion", "atc_ddd"). Plain-word names must follow the word "table"
# (a bare "reference" or "doid" is ordinary language); names with an underscore are unambiguous on their own.
TABLE_NAMES = {
    'ddd': ['atc_ddd'], 'identifiers': ['identifier', 'inxight_drugs'],
    'structure_info': ['structure_type', 'struct2parent', 'parentmol'], 'pk_measurements': ['pk_az'],
    'veterinary': ['humanim', 'vetomop', 'vetprod', 'vetprod2struct', 'vetprod_type', 'vettype'],
    'adverse_events_male': ['faers_male'], 'adverse_events_female': ['faers_female'], 'adverse_events_pediatric': ['faers_ped'],
    'adverse_events_geriatric': ['faers_ger'], 'adverse_events_top': ['faers_top'],
    'patent_details': ['ob_patent_view', 'ob_exclusivity_view'], 'doid_mapping': ['omop_relationship_doid_view'],
    'lincs': ['lincs_signature'], 'target_profile': ['target_dictionary', 'td2tc', 'target_component', 'tdgo2tc', 'tdkey2tc'],
    'ontology': ['target_go', 'target_keyword'], 'disease_profile': ['doid', 'doid_xref'], 'references': ['reference'],
    'dictionary': ['action_type', 'approval_type', 'attr_type', 'data_source', 'dbversion', 'ddi_risk', 'id_type', 'protein_type',
                   'ref_type', 'struct_type_def', 'target_class', 'ob_patent_use_code', 'ob_exclusivity_code', 'rdkit_build_meta'],
}
for _section, _tables in TABLE_NAMES.items():
    _alternatives = [r'table\s+(?:public\.)?' + t for t in _tables] + [t for t in _tables if '_' in t or any(c.isdigit() for c in t)]
    SECTION_CUES[_section] = SECTION_CUES[_section][:-3] + '|' + '|'.join(_alternatives) + r')\b' if SECTION_CUES[_section].endswith(r')\b') \
        else SECTION_CUES[_section] + r'|\b(?:' + '|'.join(_alternatives) + r')\b'


# Sections whose documents describe an entity (target, disease, reference, ontology term, code table) and carry no drug name.
ENTITY_SECTIONS = {'target_profile', 'ontology', 'disease_profile', 'references', 'dictionary'}


QUOTED = re.compile(r'"[^"]{3,}"|\u201c[^\u201d]{3,}\u201d')


def requested_sections(question):
    # Text inside quotation marks is a literal title or name (a reference title, a GO term), not part of the user's intent:
    # "Nicotinic acid: pharmacological effects and mechanisms of action" must not request the pharmacology section.
    question = QUOTED.sub(' ', question)
    return [section for section, pattern in SECTION_CUES.items() if re.search(pattern, question, re.I)]


def lexical_sections(index, drug, question, limit=4):
    query = tokens(question) - tokens(drug)
    scored = []
    for section, facts in index.sections.get(drug, {}).items():
        score = max((len(query & tokens(label + ' ' + fact)) for label, fact in facts), default=0)
        if score:
            scored.append((score, section))
    return [section for _, section in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]]


def question_evidence(index, question):
    """Prefetch named sections for unambiguous mentions; preserve missing evidence."""
    sections = requested_sections(question)
    drugs = []
    for mention in index.mentions(question):
        candidates = index.candidates(mention)
        if len(candidates) != 1:
            return []
        if candidates[0] not in drugs:
            drugs.append(candidates[0])
    return [index.evidence(drug, section, limit=30) for drug in drugs[:4] for section in sections[:4]]


def filter_records(index, filters, limit=20):
    """AND exact-token conditions over source facts, with a complete match count."""
    valid = set(SECTION_CUES) | {'properties', 'structures'}
    if not 1 <= len(filters) <= 5 or not 1 <= limit <= 50:
        raise ValueError('Provide 1–5 filters and a limit of 1–50')
    for item in filters:
        if set(item) != {'section', 'term'} or item['section'] not in valid or not tokens(item['term']):
            raise ValueError('Each filter requires a valid section and nonempty term')
    matches = []
    for drug in sorted(index.sections):
        evidence = []
        for item in filters:
            wanted = tokens(item['term'])
            facts = [(label, fact) for label, fact in index.sections[drug].get(item['section'], [])
                     if wanted <= tokens(fact)]
            if not facts:
                break
            evidence.append({'section': item['section'], 'facts': facts})
        else:
            matches.append({'drug': drug, 'evidence': evidence,
                            'url': index.evidence(drug, filters[0]['section'])['url']})
    return {'scope': 'All filters must match whole tokens within a source fact; not semantic or exhaustive clinical matching',
            'total_matches': len(matches), 'returned_matches': min(limit, len(matches)),
            'truncated': len(matches) > limit, 'matches': matches[:limit]}


def structured_section_records(index, drug, question):
    """Keep explicitly requested sections ahead of vector-only candidate discovery."""
    if not drug:
        return []
    records = []
    for section in requested_sections(question):
        evidence = index.evidence(drug, section, limit=30)
        # Named events/targets may be deep in a section; rank complete facts
        # before limiting instead of always selecting the first stored rows.
        from chatbot.retrieval import NAME_STOPWORDS
        wanted = tokens(question) - tokens(drug) - NAME_STOPWORDS
        facts = index.sections.get(drug, {}).get(section, [])
        ranked = sorted(facts, key=lambda item: -len(wanted & tokens(item[1])))
        if ranked and ranked != facts:
            from collections import defaultdict
            groups = defaultdict(list)
            for header, fact in ranked[:30]:
                groups[header].append(fact)
            evidence['text'] = '\n\n'.join(header + ':\n' + '\n'.join('- ' + fact for fact in values)
                                             for header, values in groups.items())
        if evidence['text']:
            records.append({'text': evidence['text'], 'metadata': {
                'drug_name': drug, 'struct_id': evidence['struct_id'],
                'section': section, 'dataset_version': evidence['dataset_version'],
                'truncated': evidence['truncated'],
            }})
    return records
