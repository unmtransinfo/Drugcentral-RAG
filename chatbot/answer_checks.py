"""Post-generation verification of answers against retrieved evidence.

A citation asserts provenance; it does not verify that a claim is supported.
These deterministic checks detect unsupported numeric claims, omitted
qualifications, partial lists presented without disclosure, and facts stated
for sections that have no source evidence. The active response flow rejects answers with violations. Passing these checks is NOT
proof of clinical correctness; it only confirms consistency with retrieved text.
"""
import re

UNIT_CLAIM = re.compile(
    r'\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|kg|ml|l|mmol|mol|nm|µm|um|%|percent'
    r'|hours?|minutes?|days?|weeks?|months?|years?|fold)\b', re.I)
PARTIAL_LANGUAGE = re.compile(
    r'(?i)\b(partial|first \d+|not (?:a )?complete|only (?:the )?(?:first|top)'
    r'|truncated|among the|subset|shown|not be exhaustive|selected label excerpt)\b')
FAERS_CAVEAT = re.compile(
    r'(?i)not establish (?:causation|frequency)|cannot (?:establish|determine)'
    r'|do not (?:establish|prove)|reported (?:association|events?) (?:only|do)')
SAFETY_CLAIM = re.compile(
    r'(?i)\b(safe to (?:take|use|combine)|safe together|are safe'
    r'|no (?:known )?interactions? (?:between|with|exist|are))\b')
BULLET = re.compile(r'(?:^|\n)\s*[-•]\s+\S')


# Label sections that answer the same requested field as their snapshot counterpart.
_LABEL_SECTION_EQUIVALENTS = {'adverse_reactions': 'adverse_events'}


def _searchable(evidence, observations):
    parts = [item.get('text', '') for item in evidence or [] if item.get('text')]
    parts.extend(str(observation) for observation in observations or ())
    return ' '.join(parts).lower()


def check_answer(question, answer, evidence, observations=()):
    """Return a list of {kind, detail} violations; an empty list passes.

    evidence: prefetched EvidenceIndex.evidence items supplied with the question.
    observations: raw tool outputs the agent actually used (JSON strings).
    """
    corpus = _searchable(evidence, observations)
    violations = []
    if corpus.strip():
        collapsed = re.sub(r'\s+', '', corpus)
        for value in sorted(set(UNIT_CLAIM.findall(answer))):
            if re.sub(r'\s+', '', value.lower()) not in collapsed:
                violations.append({'kind': 'unsupported_claim',
                                   'detail': f'Numeric claim "{value.strip()}" does not appear in the retrieved evidence.'})
    truncated = (any(item.get('truncated') for item in evidence or [])
                 or '"truncated": true' in corpus)
    if truncated and not PARTIAL_LANGUAGE.search(answer):
        violations.append({'kind': 'partial_list',
                           'detail': 'The evidence is truncated; the answer must disclose that it is a partial list.'})
    if 'faers' in corpus and not FAERS_CAVEAT.search(answer):
        violations.append({'kind': 'omitted_qualification',
                           'detail': 'FAERS statistics are cited; state that they do not establish causation or event frequency.'})
    if SAFETY_CLAIM.search(answer) and 'safe' not in corpus and 'no known interaction' not in corpus:
        violations.append({'kind': 'unsupported_claim',
                           'detail': 'The answer asserts safety that the retrieved evidence does not state.'})
    if evidence and not any(item.get('text') for item in evidence) and BULLET.search(answer):
        violations.append({'kind': 'missing_evidence',
                           'detail': 'The requested sections have no source evidence, but the answer lists facts.'})
    return violations


def missing_requested_fields(question, claims, evidence, drugs):
    """Find explicit requested fields absent from validated claim text per drug.

    This checks common DrugCentral fields, not arbitrary natural-language intent.
    Quotes are deliberately excluded: quoting a full record does not answer all
    of its fields. Unknown or implicit requests remain outside this check.
    """
    from chatbot.question_retrieval import SECTION_CUES
    fields = dict(SECTION_CUES)
    fields.update({
        'molecular formula': r'\b(?:molecular |chemical )?formula\b',
        'molecular weight': r'\b(?:molecular (?:weight|mass)|weight)\b',
        'gene symbol': r'\bgene symbols?\b',
        'CAS number': r'\bcas(?: registry)?(?: number)?\b',
        'half-life': r'\bhalf[ -]life\b',
        'bioavailability': r'\bbioavailability\b',
        'clearance': r'\bclearance\b',
    })
    requested = {name: pattern for name, pattern in fields.items()
                 if re.search(pattern, question, re.I)}
    if any(name in requested for name in ('molecular formula', 'molecular weight', 'CAS number')):
        requested.pop('properties', None)
    # ADME is a broad category; report the specific requested properties instead.
    if any(name in requested for name in ('half-life', 'bioavailability', 'clearance')):
        requested.pop('adme', None)
    missing = []
    for drug in drugs:
        drug_claims = [(text, ident) for text, ident, _ in claims if evidence[ident]['drug'] == drug]
        if not drug_claims:
            continue  # The existing uncovered-drug disclosure handles this.
        text = ' '.join(text for text, _ in drug_claims)
        sections = {evidence[ident].get('section') for _, ident in drug_claims}
        # Label-derived sections answer the same field as their snapshot counterpart.
        sections |= {_LABEL_SECTION_EQUIVALENTS[s] for s in list(sections) if s in _LABEL_SECTION_EQUIVALENTS}
        for name, pattern in requested.items():
            if re.search(pattern, text, re.I):
                continue
            # A claim backed by a record from the requested section answers the
            # field even when the paraphrase omits the section keyword itself
            # (e.g. "is indicated for" never says the word "indications").
            if name in SECTION_CUES and name in sections:
                continue
            missing.append((drug, name))
    return missing
