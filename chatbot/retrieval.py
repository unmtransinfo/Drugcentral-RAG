"""Deterministic, model-free DrugCentral evidence indexing and retrieval."""
import json
from chatbot.evaluation_trace import record_trace
from chatbot.answer_presentation import section_intro
import re
from collections import defaultdict
from difflib import get_close_matches
from functools import lru_cache

NAME_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "into", "onto", "over", "under", "as", "is",
    "are", "was", "were", "be", "been", "being", "am",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "there",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "this", "that", "these", "those", "such", "same", "other", "another",
    "some", "any", "all", "none", "each", "every", "few", "more", "most",
    "less", "least", "much", "many",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "use", "used", "uses", "using",
    "take", "takes", "took", "taken", "taking",
    "make", "made", "give", "gave", "given", "get", "got",
    "tell", "told", "show", "shown", "say", "said",
    "go", "went", "come", "came", "see", "saw", "know", "knew", "think",
    "find", "found", "want", "need", "feel", "felt", "work", "look",
    "good", "bad", "best", "worst", "well", "fast", "slow", "easy", "hard",
    "new", "old", "high", "low", "long", "short", "big", "small", "large",
    "true", "false", "also", "very", "even", "only", "just", "still",
    "yet", "ever", "now", "then", "soon", "after", "before", "during",
    "while", "next", "first", "last", "again",
    "drug", "drugs", "patient", "patients", "dose", "doses", "dosage",
    "treatment", "treated", "therapy", "therapies",
    "disease", "diseases", "condition", "conditions",
}

SECTION_HEADERS = {
    'Also known as': 'synonyms', 'Commercial product names': 'products', 'ATC Classifications': 'atc',
    'pKa Values': 'adme', 'ADME / Pharmacokinetics Properties': 'adme',
    'Approved Indications': 'indications', 'Indications': 'indications',
    'Off-label Uses': 'indications', 'Symptomatic Treatment': 'indications',
    'Risk Reduction': 'indications', 'Contraindications': 'contraindications',
    'Biological Targets': 'targets', 'Mechanism of Action Targets': 'targets',
    'Action Types': 'pharmacology', 'Pharmacologic Class (EPC)': 'pharmacology',
    'Mechanism of Action (MoA)': 'pharmacology', 'Pharmacologic Action': 'pharmacology',
    'Chemical Structure Class': 'pharmacology', 'Regulatory Approvals': 'approvals',
    'FDA Patents': 'patents', 'FDA Exclusivities': 'patents',
    'Protein Data Bank (PDB) Co-crystallized Codes': 'structures',
    'Clinical Adverse Events (FAERS Statistics)': 'adverse_events',
    'Drug-Drug Interactions': 'interactions',
    # Full-database coverage (ingestion/full_coverage.py): new sections only, never mixed into the ones above.
    'Defined Daily Doses (ATC DDD)': 'ddd',
    'Chemical Identifiers (all identifier types)': 'identifiers',
    'Inxight Drugs Cross-reference': 'identifiers',
    'Structure Types': 'structure_info',
    'Parent Compound': 'structure_info',
    'AstraZeneca Pharmacokinetic Measurements': 'pk_measurements',
    'Human and Animal Use': 'veterinary',
    'Veterinary Indications and Contraindications': 'veterinary',
    'Veterinary Products': 'veterinary',
    'FAERS Reports in Male Patients': 'adverse_events_male',
    'FAERS Reports in Female Patients': 'adverse_events_female',
    'FAERS Reports in Pediatric Patients': 'adverse_events_pediatric',
    'FAERS Reports in Geriatric Patients': 'adverse_events_geriatric',
    'Highest-ranked FAERS Signals': 'adverse_events_top',
    'Orange Book Patents with Use Descriptions': 'patent_details',
    'Orange Book Exclusivities with Descriptions': 'patent_details',
    'SNOMED and Disease Ontology Mappings': 'doid_mapping',
    'LINCS Signature Similarity': 'lincs',
    'Target Profile': 'target_profile',
    'Target Details': 'target_profile',
    'Protein Components': 'target_profile',
    'Gene Ontology Terms': 'target_profile',
    'Target Keywords': 'target_profile',
    'Drugs with Recorded Activity': 'target_profile',
    'Disease Profile': 'disease_profile',
    'Disease Details': 'disease_profile',
    'Cross-references': 'disease_profile',
    'Drugs Mapped to this Disease': 'disease_profile',
    'Gene Ontology Term': 'ontology',
    'Target Keyword': 'ontology',
    'Term Details': 'ontology',
    'Reference': 'references',
    'Citation': 'references',
    'Dictionary Code': 'dictionary',
    'Code Details': 'dictionary',
    'Dictionary Table': 'dictionary',
    'Rows': 'dictionary',
}
EMPTY = {'none documented', 'none reported', 'n/a', 'unknown', 'not available', 'none'}
MISSING = 'The requested information is not available in the loaded DrugCentral snapshot.'


def tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def section_blocks(text):
    """Split at semantic headers, including scalar properties."""
    header, lines = 'Profile', []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('-') and ':' in stripped:
            if lines:
                yield header, '\n'.join(lines)
            label, value = stripped.split(':', 1)
            header, lines = label, ([f'- {value.strip()}'] if value.strip() else [])
        else:
            lines.append(line)
    if lines:
        yield header, '\n'.join(lines)


def section_chunks(text, max_chars=3000):
    """Keep section identity on every chunk; never split a source fact."""
    for header, body in section_blocks(text):
        section = SECTION_HEADERS.get(header, 'properties')
        current, size, sequence = [], len(header) + 2, 0
        for line in body.splitlines():
            if current and size + len(line) + 1 > max_chars:
                yield section, header, sequence, header + ':\n' + '\n'.join(current)
                current, size, sequence = [], len(header) + 2, sequence + 1
            current.append(line)
            size += len(line) + 1
        if current:
            yield section, header, sequence, header + ':\n' + '\n'.join(current)


class EvidenceIndex:
    def __init__(self, data):
        self.sections = defaultdict(lambda: defaultdict(list))
        self.ids, self.names = {}, defaultdict(set)
        self.ingredient_names = defaultdict(set)
        self.versions = set()
        self.legacy = set()
        grouped = defaultdict(list)
        seen_facts = defaultdict(set)
        for doc, meta in zip(data['documents'], data['metadatas']):
            drug = str(meta.get('drug_name', '')).lower().strip()
            if not drug:
                continue
            grouped[drug].append((doc, meta))
            for field, target in [('ingredient_synonyms_json', self.ingredient_names),
                                  ('product_names_json', self.names)]:
                for name in json.loads(meta.get(field, '[]')):
                    if name.strip():
                        target[name.strip().lower()].add(drug)
            if meta.get('struct_id') is not None:
                self.ids[drug] = int(meta['struct_id'])
            self.versions.add(str(meta.get('dataset_version', 'legacy')))
        for drug, rows in grouped.items():
            self.names[drug].add(drug)
            if all(meta.get('schema_version') == 2 for _, meta in rows):
                # Global chunk_index ordering comes from storage.
                documents = [doc for doc, _ in rows]
            else:
                self.legacy.add(drug)
                # Legacy split_dossier inserted a profile prefix but retained all bytes.
                documents = [''.join(re.sub(r'^Drug Profile: .*? \(Continued\.\.\.\)\n', '', doc)
                                     for doc, _ in rows)]
            for doc in documents:
                for header, body in section_blocks(doc):
                    section = SECTION_HEADERS.get(header, 'properties')
                    facts = [line.strip()[2:].strip() for line in body.splitlines()
                             if line.strip().startswith('- ')]
                    facts = [fact for fact in facts if fact.lower() not in EMPTY]
                    if not facts:
                        continue
                    # Old collections mixed relationship types; never imply approval.
                    label = 'Reported uses (relationship type unavailable)' if header == 'Approved Indications' else header
                    for fact in facts:
                        item = (label, fact)
                        if item not in seen_facts[(drug, section)]:
                            seen_facts[(drug, section)].add(item)
                            self.sections[drug][section].append(item)
                    if section in {'synonyms', 'products'}:
                        for name in facts:
                            if len(name) >= 4 and name.lower() not in NAME_STOPWORDS:
                                self.names[name.lower()].add(drug)
        for name, drugs in self.ingredient_names.items():
            self.names[name] = set(drugs)
        if 'acetylsalicylic acid' in self.sections:
            self.names['aspirin'] = {'acetylsalicylic acid'}
        self.pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(n) for n in sorted(self.names, key=lambda n: (-len(n), n))) + r')(?!\w)', re.I) if self.names else None
        self.condition_index = defaultdict(set)
        self.condition_facts = []
        for drug, sections in self.sections.items():
            for label, fact in sections.get('indications', []):
                idx = len(self.condition_facts)
                self.condition_facts.append((drug, label, fact))
                for token in tokens(fact):
                    self.condition_index[token].add(idx)

    def candidates(self, name):
        name = name.lower().strip()
        if name in self.sections:
            return [name]
        return sorted(self.names.get(name, set()))

    def mentions(self, question):
        if not self.pattern:
            return []
        # A product alias may also be a section term. Ignore "target" only
        # in explicit target-question phrases; keep standalone product names
        # resolvable and preserve ambiguity for actual drug mentions.
        section_spans = [m.span() for m in re.finditer(
            r"\b(?:biological|molecular)\s+targets?\b|"
            r"\btargets?\s+(?:(?:recorded|listed)\s+)?(?:of|for)\b|"
            r"\b(?:associated\s+with|bind\s+to|act\s+on)\s+"
            r"(?:the\s+)?(?:human\s+)?[a-z][a-z0-9-]*\s+targets?\b|"
            # "target class / keyword / profile ...": target is the noun of the question, not a product name.
            r"\btargets?\s+(?:keywords?|class(?:es)?|profiles?|components?|development\s+level|accessions?|annotations?)\b|"
            # "... of / for the CACNA2D1 target" (a target named by gene or protein, up to four words).
            r"\b(?:of|for|on|about)\s+the\s+[a-z0-9][a-z0-9-]*(?:\s+[a-z0-9-]+){0,3}\s+targets?\b",
            question, re.I,
        )]
        event_spans = [m.span() for m in re.finditer(r'\bFAERS\s+event\s+[^?]+', question, re.I)]
        mentions = []
        for match in self.pattern.finditer(question):
            name = match.group(0).lower()
            if any(start <= match.start() and match.end() <= end for start, end in event_spans):
                continue
            if name in {'target', 'targets'} and any(
                start <= match.start() and match.end() <= end
                for start, end in section_spans
            ):
                continue
            mentions.append(name)
        return list(dict.fromkeys(mentions))

    def suggestions(self, name):
        return get_close_matches(name.lower().strip(), sorted(self.names), n=3, cutoff=.82)

    @lru_cache(maxsize=2048)
    def lookup(self, drug, section):
        groups = defaultdict(list)
        for label, fact in self.sections.get(drug, {}).get(section, []):
            groups[label].append(fact)
        return '\n\n'.join(label + ':\n' + '\n'.join('- ' + f for f in facts)
                            for label, facts in groups.items())

    def search_condition(self, condition):
        query = tokens(condition)
        if not query:
            return []
        matches = set.intersection(*(self.condition_index.get(t, set()) for t in query))
        return sorted({self.condition_facts[i] for i in matches})

    def evidence(self, drug, section, limit=None):
        facts = self.sections.get(drug, {}).get(section, [])
        selected = facts if limit is None else facts[:limit]
        groups = defaultdict(list)
        for label, fact in selected:
            groups[label].append(fact)
        text = '\n\n'.join(label + ':\n' + '\n'.join('- ' + fact for fact in values)
                            for label, values in groups.items())
        return {'drug': drug, 'struct_id': self.ids.get(drug), 'section': section,
                'dataset_version': ','.join(sorted(self.versions)), 'text': text,
                'total_facts': len(facts), 'returned_facts': len(selected),
                'truncated': len(selected) < len(facts),
                'url': f'https://drugcentral.org/drugcard/{self.ids[drug]}' if drug in self.ids else None}


# Only complete, simple question forms bypass the LLM. Qualifiers go to the agent.
TARGET_PATTERNS = [
    re.compile(r'(?:what|which) (?:biological |molecular )?targets? (?:are associated with|does) (.+?)(?: act on)?[?.!]*$', re.I),
    re.compile(r'(?:what are |list |show )?(?:the )?(?:biological |molecular )?targets? (?:of|for) (.+?)[?.!]*$', re.I),
]


def source_target_answer(index, question):
    """Render only unqualified target lists; a qualifier is never treated as a drug."""
    for pattern in TARGET_PATTERNS:
        match = pattern.fullmatch(question.strip())
        if match and index.candidates(match.group(1)):
            return direct_answer(index, 'What are targets of ' + match.group(1) + '?')
    return None


INDICATION_PATTERNS = [
    re.compile(r'(?:what|which) indications? (?:are associated with|are recorded for|does) (.+?)(?: have)?[?.!]*$', re.I),
    re.compile(r'(?:what are|list|show)(?: me)? (?:the )?indications? (?:of|for) (.+?)[?.!]*$', re.I),
    re.compile(r'what (?:is|are) (.+?) used for[?.!]*$', re.I),
]


def source_indication_answer(index, question):
    """Unqualified indication lists preserve source relationship labels verbatim."""
    question = re.sub(r'\s+in DrugCentral[?.!]*$', '', question.strip(), flags=re.I)
    for pattern in INDICATION_PATTERNS:
        match = pattern.fullmatch(question)
        if not match:
            continue
        candidates = index.candidates(match[1].strip())
        if len(candidates) != 1:
            return None
        item = index.evidence(candidates[0], 'indications', limit=30)
        if not item['text']:
            return None
        answer = f"DrugCentral records the following uses for **{candidates[0]}**:\n\n" + item['text']
        if item['truncated']:
            answer += f"\n\nPartial list: {item['returned_facts']} of {item['total_facts']} recorded facts."
        answer += '\n\nIndications, off-label uses, and other relationships retain their source labels.'
        if item['url']:
            answer += f"\n\nSource: [DrugCentral — {candidates[0]}]({item['url']})."
        record_trace('direct-indications', [item])
        return answer
    return None


DIRECT_PATTERNS = [
    (re.compile(r'what (?:is|are) (.+?) used for[?.!]*$', re.I), 'indications'),
    (re.compile(r'(?:what are |list |show )?(?:the )?(indications|contraindications|targets|interactions|side effects|adverse events|synonyms|brand names|approvals|atc classification|mechanism of action|adme properties|patents) (?:of|for) (.+?)[?.!]*$', re.I), None),
]
INTENTS = {'side effects': 'adverse_events', 'brand names': 'synonyms', 'atc classification': 'atc',
           'mechanism of action': 'pharmacology', 'adme properties': 'adme'}


def direct_answer(index, question):
    for pattern, section in DIRECT_PATTERNS:
        match = pattern.fullmatch(question.strip())
        if not match:
            continue
        name = match.group(1) if section else match.group(2)
        section = section or INTENTS.get(match.group(1).lower(), match.group(1).lower())
        candidates = index.candidates(name)
        if len(candidates) > 1:
            return 'This name maps to multiple ingredients or drugs: ' + ', '.join(candidates) + '. Please specify which one.'
        if not candidates:
            suggestions = index.suggestions(name)
            if index.mentions(name):
                return None
            return ('Did you mean ' + ', '.join(suggestions) + '? Please confirm the drug name.') if suggestions else MISSING
        drug = candidates[0]
        evidence = index.evidence(drug, section)
        record_trace('direct', [evidence])
        if not evidence['text']:
            return MISSING
        answer = section_intro(drug) + '\n\n' + evidence['text']
        if section == 'adverse_events':
            answer += '\n\nFAERS reports do not establish causation or how frequently an event occurs.'
        if evidence['url']:
            answer += f'\n\nSource: [DrugCentral — {drug.title()}]({evidence["url"]}) ({section}).'
        return answer
    return None



def source_grounded_answer(index, question):
    """Source-rendered answer for any simple, unqualified single-drug lookup.

    Covers target lists and the DIRECT_PATTERNS section lookups. Source labels
    (e.g. Off-label Uses) and citations are preserved verbatim; qualified or
    multiple-drug questions return None and stay with the agent.
    """
    from chatbot.source_queries import factual_lookup_answer
    answer = factual_lookup_answer(index, question)
    if answer is not None:
        return answer
    record_trace('direct')
    answer = source_target_answer(index, question)
    return answer if answer is not None else direct_answer(index, question)


_FAERS_CASES = re.compile(r'\((?:LLR:\s*[\d.]+\s*,\s*)?([\d,]+)\s+cases?\)', re.I)


def _bullet_facts(text):
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith('- ')]


def grounded_fallback(index, question, evidence, *, limitation=None, faers_top=12):
    """Render prefetched evidence directly when generation fails verification.

    Guarantees every statement is a source fact with its label and citation.
    Records for the same drug and field merge into a single list; FAERS lists
    are sorted by report count and capped, with omissions disclosed. Returns
    None when there is no usable evidence to render.
    """
    usable = [item for item in evidence if item.get('text')]
    if not usable:
        return None
    groups = {}
    for item in usable:
        kind = 'product_label' if item.get('source_type') == 'product_label' else 'record'
        key = (item['drug'], item.get('section', 'retrieved records'), kind)
        group = groups.setdefault(key, {'item': item, 'texts': [], 'truncated': False,
                                        'returned_facts': None, 'total_facts': None})
        group['texts'].append(item['text'])
        group['truncated'] = group['truncated'] or bool(item.get('truncated'))
        if item.get('returned_facts') is not None and item.get('total_facts') is not None:
            group['returned_facts'] = (group['returned_facts'] or 0) + item['returned_facts']
            group['total_facts'] = max(group['total_facts'] or 0, item['total_facts'])
    parts = []
    faers_shown = False
    for (drug, section, kind), group in groups.items():
        item = group['item']
        facts = []
        for text in group['texts']:
            for fact in _bullet_facts(text):
                if fact not in facts:
                    facts.append(fact)
        is_faers = section == 'adverse_events' and any(_FAERS_CASES.search(fact) for fact in facts)
        omitted = 0
        if is_faers:
            facts.sort(key=lambda fact: int(_FAERS_CASES.search(fact).group(1).replace(',', ''))
                       if _FAERS_CASES.search(fact) else -1, reverse=True)
            if len(facts) > faers_top:
                omitted = len(facts) - faers_top
                facts = facts[:faers_top]
        body = '\n'.join('- ' + fact for fact in facts) if facts else '\n\n'.join(group['texts'])
        source_label = ('Product label (DailyMed)' if kind == 'product_label'
                        else 'FAERS adverse event reports' if section == 'adverse_events'
                        else 'DrugCentral structured record')
        block = f"**{drug.title()} — {section.replace('_', ' ')}** ({source_label}):\n\n{body}"
        returned, total = group['returned_facts'], group['total_facts']
        if (returned is None or total is None) and facts and drug in index.sections:
            index_total = index.evidence(drug, section)['total_facts']
            if index_total and index_total >= len(facts):
                returned, total = len(facts), index_total
        if omitted:
            block += f"\n\nPartial list: showing {len(facts)} of {len(facts) + omitted} source facts."
        elif group['truncated']:
            if returned is not None and total is not None:
                block += f"\n\nPartial list: showing {returned} of {total} source facts."
            else:
                block += '\n\nPartial list: the retrieved record is an excerpt of the source text.'
        if item.get('url'):
            name = item.get('product_name') or f"DrugCentral — {drug.title()}"
            block += f"\n\nSource: [{name}]({item['url']}) ({section})."
        parts.append(block)
        faers_shown = faers_shown or section == 'adverse_events'
    header = ('A generated summary could not be verified against the evidence, '
              'so the verified source facts are listed directly.')
    if limitation:
        header += f"\n\n{limitation}"
    answer = header + '\n\n' + '\n\n'.join(parts)
    if faers_shown:
        answer += '\n\nFAERS reports do not establish causation or how frequently an event occurs.'
    return answer