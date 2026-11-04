"""Conservative, source-only answers for explicit factual lookup requests.

Only recognized question forms are handled. Clinical advice, comparisons and
qualifiers outside this grammar require clarification in the strict answer path. No benchmark data is used.
"""
import re
from chatbot.evaluation_trace import record_trace
from chatbot.answer_presentation import single_fact_intro

# Keep related source relationships separate (e.g. indications vs off-label uses).
FIELDS = {
    r'(?:biological |molecular )?target': ('targets', 'Biological Targets'),
    r'indication': ('indications', 'Indications'),
    r'contraindication': ('contraindications', 'Contraindications'),
    r'regulatory (?:agency and (?:its )?(?:recorded )?approval year|approval)': ('approvals', 'Regulatory Approvals'),
    r'ADME property': ('adme', 'ADME / Pharmacokinetics Properties'),
    r'(?:established )?pharmacologic class': ('pharmacology', 'Pharmacologic Class (EPC)'),
    r'ATC code(?: and (?:its )?(?:associated )?description)?': ('atc', 'ATC Classifications'),
    r'PDB (?:co-crystallized )?(?:structure )?code': ('structures', 'Protein Data Bank (PDB) Co-crystallized Codes'),
}
PREFIX = r'(?:please\s+)?(?:name|give|show|list)\s+(?:me\s+)?(?:one|an?|a single)\s+(?:recorded\s+)?'
SUFFIX = (r'(?: in (?:this|the) DrugCentral snapshot)?'
          r'(?:; include its gene symbol if recorded|, including its value and units if present)?[?.!]*')
LOOKUPS = [(re.compile(PREFIX + field + r'\s+(?:(?:recorded|listed)\s+)?(?:for|of)\s+(?P<drug>.+?)' + SUFFIX, re.I), section, header)
           for field, (section, header) in FIELDS.items()]
PAIR = re.compile(r'(?:what|which) (?:is the )?interaction severity (?:is recorded )?between (?P<first>.+?) and (?P<second>.+?)[?.!]*', re.I)
EVENT = re.compile(r'for (?P<drug>.+?),\s*what (?:LLR and case count|case count and LLR) (?:are recorded )?for (?:the )?FAERS event (?P<event>[^?]+)\?(?:\s*Describe this as a report association, not proof of causation\.)?', re.I)


def resolve(index, name):
    candidates = index.candidates(name.strip())
    return candidates[0] if len(candidates) == 1 else None


def render(index, drug, section, facts):
    evidence = index.evidence(drug, section)
    # Record exactly the selected facts that support the answer, not an arbitrary
    # prefix of a long section. Selection searches the complete source section.
    selected = facts[:1]
    evidence.update(text='\n'.join(label + ':\n- ' + fact for label, fact in selected),
                    returned_facts=len(selected), truncated=len(selected) < evidence['total_facts'])
    record_trace('direct', [evidence])
    if not selected:
        return 'The requested entry is not available in the loaded DrugCentral snapshot.'
    label, fact = selected[0]
    answer = single_fact_intro(drug, section) + f'\n\n- {fact}'
    if section == 'adverse_events':
        answer += '\n\nThis is a report association; FAERS reports do not establish causation or event frequency.'
    if evidence.get('url'):
        answer += f'\n\nSource: [DrugCentral — {drug.title()}]({evidence["url"]}).'
    return answer


def factual_lookup_answer(index, question):
    question = question.strip()
    for pattern, section, header in LOOKUPS:
        match = pattern.fullmatch(question)
        if match:
            drug = resolve(index, match['drug'])
            if drug is None:
                return None
            # Reject suffixes meant for another field rather than ignoring them.
            if '; include its gene symbol' in question.lower() and section != 'targets':
                return None
            if ', including its value and units' in question.lower() and section != 'adme':
                return None
            facts = [(label, fact) for label, fact in index.sections[drug].get(section, []) if label == header]
            return render(index, drug, section, facts)
    match = PAIR.fullmatch(question)
    if match:
        first, second = resolve(index, match['first']), resolve(index, match['second'])
        if first is None or len(index.candidates(match['second'].strip())) > 1:
            return None
        # A named interaction partner can be a class rather than an indexed drug.
        # Match the class name literally; never infer membership of a drug in it.
        partner = ' '.join(match['second'].casefold().split())
        facts = [(label, fact) for label, fact in index.sections[first].get('interactions', [])
                 if ' [' in fact and (
                     ' '.join(fact.split(' [', 1)[0].casefold().split()) == partner
                     or (second is not None and resolve(index, fact.split(' [', 1)[0]) == second))]
        if second is None and not facts:
            return None
        return render(index, first, 'interactions', facts)
    match = EVENT.fullmatch(question)
    if match:
        drug = resolve(index, match['drug'])
        if drug is None:
            return None
        event = ' '.join(match['event'].casefold().split())
        facts = [(label, fact) for label, fact in index.sections[drug].get('adverse_events', [])
                 if ' (LLR:' in fact and ' '.join(fact.split(' (LLR:', 1)[0].casefold().split()) == event]
        return render(index, drug, 'adverse_events', facts)
    return None
