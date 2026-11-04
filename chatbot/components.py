"""Question decomposition, evidence-index mapping and deterministic composition.

A multi-part question becomes independent (drug, field) components. Each component is
retrieved, generated, validated and repaired on its own, then the validated claims are
composed without another model call, so one failed component cannot remove the others.
"""
import re

from chatbot.question_retrieval import requested_sections

MAX_COMPONENTS = 8
PAIR = re.compile(r'\b(?:interact(?:s|ing)?\s+with|taken\s+together|used\s+together|combined\s+with|given\s+with|co-?administered)\b', re.I)

SECTION_LABELS = {
    'properties': 'Chemical properties', 'targets': 'Targets', 'indications': 'Indications',
    'contraindications': 'Contraindications', 'adme': 'ADME / pharmacokinetics',
    'approvals': 'Regulatory approvals', 'interactions': 'Interactions',
    'adverse_events': 'Adverse reactions', 'pharmacology': 'Pharmacology',
    'synonyms': 'Synonyms and brand names', 'atc': 'ATC classification',
    'structures': 'Protein Data Bank structures', 'patents': 'Patents and exclusivity',
    'ddd': 'Defined daily doses',
    'identifiers': 'Identifiers',
    'structure_info': 'Structure information',
    'pk_measurements': 'AstraZeneca PK measurements',
    'veterinary': 'Veterinary use',
    'adverse_events_male': 'FAERS reports (male patients)',
    'adverse_events_female': 'FAERS reports (female patients)',
    'adverse_events_pediatric': 'FAERS reports (pediatric patients)',
    'adverse_events_geriatric': 'FAERS reports (geriatric patients)',
    'adverse_events_top': 'Highest-ranked FAERS signals',
    'patent_details': 'Orange Book product details',
    'doid_mapping': 'Disease Ontology mappings',
    'lincs': 'LINCS signature similarity',
    'target_profile': 'Target profile',
    'ontology': 'Ontology terms',
    'disease_profile': 'Disease profile',
    'references': 'References',
    'dictionary': 'Dictionary entries',
}
# Each phrase must trigger exactly its own field in requested_sections(); a test enforces this.
SECTION_PHRASES = {
    'properties': 'the molecular formula and molecular weight of {drug}',
    'targets': 'the targets of {drug}',
    'indications': 'the indications of {drug}',
    'contraindications': 'the contraindications of {drug}',
    'adme': 'the half-life and bioavailability of {drug}',
    'approvals': 'the regulatory approvals of {drug}',
    'interactions': 'the drug interactions of {drug}',
    'adverse_events': 'the common adverse reactions of {drug}',
    'pharmacology': 'the mechanism of action of {drug}',
    'synonyms': 'the synonyms of {drug}',
    'atc': 'the ATC classification of {drug}',
    'structures': 'the PDB structure codes of {drug}',
    'patents': 'the patents of {drug}',
    'ddd': 'the defined daily doses (DDD) of {drug}',
    'identifiers': 'the UNII and PubChem identifiers of {drug}',
    'structure_info': 'the parent compound of {drug}',
    'pk_measurements': 'the AstraZeneca PK measurements of {drug}',
    'veterinary': 'the veterinary uses of {drug}',
    'adverse_events_male': 'the highest-LLR events of {drug} in male patients',
    'adverse_events_female': 'the highest-LLR events of {drug} in female patients',
    'adverse_events_pediatric': 'the highest-LLR events of {drug} in pediatric patients',
    'adverse_events_geriatric': 'the highest-LLR events of {drug} in geriatric patients',
    'adverse_events_top': 'the strongest adverse signals of {drug}',
    'patent_details': 'the Orange Book trade names and applicants of {drug}',
    'doid_mapping': 'the DOID, SNOMED and UMLS mappings of {drug}',
    'lincs': 'the LINCS signature similarity of {drug}',
    'target_profile': 'the UniProt accession and protein class',
    'ontology': 'the term details of GO:0000002',
    'disease_profile': 'the disease profile',
    'references': 'the PMID and DOI of the reference',
    'dictionary': 'the dictionary entries',
}


def plan_question(question, drugs, limit=MAX_COMPONENTS):
    """Deterministic planner: SIMPLE, or MULTI_PART with one component per drug and field."""
    sections = list(dict.fromkeys(requested_sections(question)))
    subjects = list(drugs) or [None]
    components = []
    for drug in subjects:
        for section in sections:
            phrase = SECTION_PHRASES[section].format(drug=drug or 'the drug')
            components.append({'id': f'c{len(components) + 1}', 'drug': drug, 'section': section,
                               'question': 'What are ' + phrase + '?'})
    # Entity questions (a target, a GO term, a reference) name no drug: decomposing them would replace the user's own words
    # with generic phrases and lose the entity, so they stay on the single path, which can search several sections at once.
    # A question about the relationship between drugs ("interaction between A and B", "A taken together with B") is about the
    # pair; splitting it into one component per drug retrieves each drug's whole list and loses the pair.
    if len(drugs) >= 2 and (sections == ['interactions'] or PAIR.search(question)):
        return {'type': 'SIMPLE', 'components': components, 'omitted': 0}
    if len(components) < 2 or not drugs:
        return {'type': 'SIMPLE', 'components': components, 'omitted': 0}
    return {'type': 'MULTI_PART', 'components': components[:limit],
            'omitted': max(0, len(components) - limit)}


def generation_evidence(evidence):
    """Model-facing evidence uses compact local indexes; the backend owns the real identifiers."""
    fields = ('drug_name', 'section', 'source_type', 'product_name', 'route', 'form', 'dataset_version')
    return [dict(index=i, text=e.text, metadata={k: e.metadata[k] for k in fields if k in e.metadata})
            for i, e in enumerate(evidence)]


def resolve_indexes(payload, evidence):
    """Map claim `evidence` indexes to immutable ids; an invalid index can never validate.

    Claims that already carry real `evidence_ids` pass through, since membership in the
    selected set is verified downstream either way.
    """
    claims = payload.get('claims') if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        return payload
    ids = [e.evidence_id for e in evidence]
    resolved = []
    for claim in claims:
        if isinstance(claim, dict) and 'evidence' in claim:
            claim = dict(claim)
            refs = claim.pop('evidence')
            valid = isinstance(refs, list) and refs and all(type(i) is int and 0 <= i < len(ids) for i in refs)
            claim['evidence_ids'] = [ids[i] for i in refs] if valid else ['invalid-evidence-index']
        resolved.append(claim)
    return {**payload, 'claims': resolved}


def citation_line(by_id, used, numbers):
    parts = []
    for eid in used:
        metadata = by_id[eid].metadata
        label = metadata.get('drug_name', 'DrugCentral')
        url = metadata.get('source_url')
        if url and str(url).startswith(('https://drugcentral.org/', 'https://dailymed.nlm.nih.gov/')):
            parts.append(f'[{numbers[eid]}] [{label}]({url})')
        else:
            parts.append(f'[{numbers[eid]}] {label} — DrugCentral source record')
    return ' · '.join(parts)


def source_notes(result, used, by_id):
    """Label scope, FAERS and truncation caveats required whenever such evidence is cited."""
    scopes = list(dict.fromkeys(
        f"{by_id[i].metadata.get('product_name')} ({by_id[i].metadata.get('route')}; {by_id[i].metadata.get('form')})"
        for i in used if by_id[i].metadata.get('source_type') == 'product_label'))
    if scopes:
        result = 'Label scope: ' + '; '.join(scopes) + '.\n\n' + result
    if any('faers' in by_id[i].text.lower() for i in used):
        result += '\n\nFAERS reports do not establish causation or event frequency.'
    if any(by_id[i].metadata.get('truncated') for i in used):
        result += '\n\nThis is a partial excerpt of the source evidence.'
    return result


def component_title(component, multi_drug):
    label = SECTION_LABELS.get(component['section'], component['section'])
    return f"{label} — {component['drug']}" if multi_drug and component['drug'] else label


def compose(results, by_id):
    """Deterministically join validated component claims; no new text is introduced.

    Returns (text, used evidence ids). Components without validated claims are named, not
    silently dropped, so a reader can see what the evidence did not support.
    """
    multi_drug = len({r['component']['drug'] for r in results}) > 1
    numbers = {}
    blocks, unsupported, clarifications = [], [], []
    for r in results:
        title = component_title(r['component'], multi_drug)
        if r['clarification']:
            clarifications.append(r['clarification'])
        if not r['kept']:
            unsupported.append(title)
            continue
        lines = []
        for claim in r['kept']:
            for eid in claim['evidence_ids']:
                numbers.setdefault(eid, len(numbers) + 1)
            lines.append('- ' + claim['text'] + ' ' + ''.join(f'[{numbers[i]}]' for i in claim['evidence_ids']))
        blocks.append(f'**{title}**\n' + '\n'.join(lines))
    text = '\n\n'.join(blocks)
    used = list(numbers)
    if used:
        text += '\n\nSources: ' + citation_line(by_id, used, numbers)
    if unsupported:
        text += '\n\nNot verified from the available DrugCentral evidence: ' + '; '.join(unsupported) + '.'
        text += '\n\nThis is a partial answer based on the available evidence.'
    if clarifications:
        text += '\n\n' + ' '.join(dict.fromkeys(clarifications))
    return text, used


def merge_counters(target, source):
    """Fold one component's metrics into the request metrics (sums; flags OR together)."""
    for key, value in source.items():
        if isinstance(value, bool):
            target[key] = bool(target.get(key)) or value
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            target.setdefault(key, value)
