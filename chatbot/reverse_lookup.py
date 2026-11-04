"""Exact cross-drug relationship lookups; do not use vector top-k as a database query."""
import os
from pathlib import Path
import re
import sqlite3
from chatbot.evaluation_trace import record_trace

INDICATION = re.compile(r'(?:which|what) (?:drugs|medications) (?:have (?:an? )?indications? for|are indicated for) (.+)',re.I)
TARGET = re.compile(r'(?:which|what) drugs are associated with (?:the )?(?:(human|mouse|rat) )?(.+?) targets?',re.I)
ORGANISMS={'human':'homo sapiens','mouse':'mus musculus','rat':'rattus norvegicus'}


def reverse_answer(index,question,path=None,limit=20,fallback_on_missing=False):
    question=re.sub(r'\s+in drugcentral\s*[?.!]*$', '', question.strip(),flags=re.I).rstrip('?.!')
    match=INDICATION.fullmatch(question)
    if match:
        term=match[1].strip()
        # Do not promote off-label, preventive, or legacy unknown relationships.
        rows=[(d,h,f) for d,h,f in index.search_condition(term) if h=='Indications']
        return None if fallback_on_missing and not rows else render(index,rows, f'an indication matching **{term}**', limit)
    match=TARGET.fullmatch(question)
    if not match:return None
    species,target=match.groups();target=target.strip()
    path=Path(path or os.getenv('RELATIONS_DATABASE', str(Path(__file__).resolve().parents[1]/'data/relations.sqlite')))
    if not path.exists():
        if fallback_on_missing:
            return None
        record_trace('abstain',reason='target-relations-unavailable')
        return 'The organism-aware target index is unavailable. I cannot verify this target association from the current text extracts.'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as c:
        # Exact symbol or full target-name matching; EGFR must not match EGFR2.
        rows=c.execute('''SELECT d.name,t.name,a.gene,a.organism,d.id FROM associations a
            JOIN drugs d ON d.id=a.drug_id JOIN targets t ON t.id=a.target_id
            WHERE (? IS NULL OR lower(a.organism)=?)''',
            (species,ORGANISMS.get(species.lower()) if species else None)).fetchall()
    facts=[]
    for drug,name,gene,organism,ident in rows:
        if (target.casefold()!=name.casefold() and target.casefold() not in
                {g.strip().casefold() for g in re.split(r'[|;,]',gene or '')}):continue
        # Restrict source relations to the loaded drug identities.
        drug=drug.lower().strip()
        if index.ids.get(drug)!=ident:continue
        facts.append((drug,'Target association',f'{name} ({gene or "gene unavailable"}); organism: {organism or "not recorded"}'))
    scope=f'an association with the {species+" " if species else ""}**{target}** target'
    return None if fallback_on_missing and not facts else render(index,sorted(set(facts)),scope,limit)


def render(index,rows,scope,limit):
    rows=sorted(set(rows));drug_count=len({r[0] for r in rows})
    if not rows:
        record_trace('direct-reverse',[])
        return f'No matching records for {scope} were found in the loaded snapshot. This is not proof that no such relationship exists.'
    parts=[f'DrugCentral records {drug_count} drugs with {scope}:'];evidence=[]
    for drug,relation,fact in rows[:limit]:
        url=f'https://drugcentral.org/drugcard/{index.ids[drug]}' if drug in index.ids else None
        label=f'[{drug}]({url})' if url else drug
        parts.append(f'- {label} — {relation}: {fact}')
        evidence.append(dict(drug=drug,text=relation+': '+fact,url=url,section='indications' if relation=='Indications' else 'targets'))
    if len(rows)>limit:parts.append(f'Partial list: showing {limit} of {len(rows)} matching relationship records.')
    parts.append('These are recorded database relationships, not treatment recommendations.')
    record_trace('direct-reverse',evidence)
    return '\n\n'.join([parts[0],'\n'.join(parts[1:])])
