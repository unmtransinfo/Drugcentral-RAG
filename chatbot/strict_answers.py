"""Deterministic source lookup helpers for exact facts and condition lists.

No language model is called. Unrecognized requests are not treated as proof that
information is absent from the database.
"""
import re
from chatbot.evaluation_trace import record_trace
from chatbot.retrieval import source_grounded_answer
from chatbot.source_queries import resolve, render

UNMATCHED = ('I could not reliably match this question to a supported DrugCentral lookup. '
             'Please name the drug and the exact fact or section you want. '
             'This does not mean the information is absent from the database.')
PROPERTY = re.compile(r"(?:what is|show|give)(?: me)? (?:the )?(?P<property>half-life|bioavailability|clearance|volume of distribution|water solubility) (?:of|for) (?P<drug>.+?)[?.!]*", re.I)


CONDITION = re.compile(
    r"(?:what are (?:the )?(?:treatments|drug treatments|medications|drugs) for|"
    r"(?:what|which) (?:drugs|medications) (?:treat|are used (?:for|to treat))|"
    r"(?:list|show)(?: me)? (?:drugs|medications|treatments) for) "
    r"(?P<condition>[\w][\w\s-]*?)[?.!]*", re.I)


def condition_answer(index, condition, limit=20):
    matches = index.search_condition(condition)
    if not matches:
        record_trace('abstain')
        return ('No recorded uses matching these condition terms were found in the loaded '
                'DrugCentral snapshot. This does not establish that no treatments exist.')
    selected = matches[:limit]
    lines = [f'DrugCentral lists the following drugs with recorded uses related to **{condition}**:', '']
    evidence = []
    for drug, label, fact in selected:
        item = index.evidence(drug, 'indications')
        item.update(text=label + ':\n- ' + fact, returned_facts=1,
                    truncated=item['total_facts'] > 1)
        evidence.append(item)
        citation = f' ([DrugCentral]({item["url"]}))' if item.get('url') else ''
        relationship = 'Recorded use' if label == 'Indications' else label
        lines.append(f'- **{drug}** — {relationship}: {fact}{citation}')
    if len(matches) > limit:
        lines.append(f'\nPartial list: showing {len(selected)} of {len(matches)} matching records.')
    lines.append('\nThese are database records, not a treatment recommendation. Off-label and preventive uses are labeled separately.')
    record_trace('direct', evidence)
    return '\n'.join(lines)


def strict_answer(index, question):
    record_trace('abstain')
    condition = CONDITION.fullmatch(question.strip())
    if condition:
        return condition_answer(index, condition['condition'])
    answer = source_grounded_answer(index, question)
    if answer is not None:
        return answer
    match = PROPERTY.fullmatch(question.strip())
    if match:
        drug = resolve(index, match['drug'])
        if drug is not None:
            wanted = match['property'].casefold()
            facts = [(label, fact) for label, fact in index.sections[drug].get('adme', [])
                     if re.sub(r'\s*\([^)]*\)', '', fact.split(':', 1)[0]).strip().casefold() == wanted]
            return render(index, drug, 'adme', facts)
    record_trace('abstain')
    mentions = index.mentions(question)
    for mention in mentions:
        candidates = index.candidates(mention)
        if len(candidates) > 1:
            return (f"'{mention}' maps to multiple ingredients or drugs: "
                    + ', '.join(candidates) + '. Please specify which one.')
    return UNMATCHED
