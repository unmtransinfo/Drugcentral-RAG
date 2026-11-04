"""Readable source-only framing; source facts themselves stay unchanged."""
DESCRIPTIONS = {
    'targets': 'biological target',
    'indications': 'indication',
    'contraindications': 'contraindication',
    'approvals': 'regulatory approval',
    'adme': 'ADME property',
    'pharmacology': 'pharmacologic class',
    'atc': 'ATC classification',
    'structures': 'PDB structure code',
    'interactions': 'interaction record',
    'adverse_events': 'FAERS report association',
}


def single_fact_intro(drug, section):
    description = DESCRIPTIONS.get(section, 'source fact')
    return f'DrugCentral records the following {description} for **{drug}**:'


def section_intro(drug):
    return f'DrugCentral records the following information for **{drug}**:'
