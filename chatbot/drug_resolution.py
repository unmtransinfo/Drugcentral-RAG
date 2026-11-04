"""Shared drug-name resolution and conservative question-name extraction."""
import re

from chatbot.retrieval import NAME_STOPWORDS, tokens


def resolve_name(index, name):
    """Return (official name, clarification); suggestions never select a drug."""
    candidates = index.candidates(name)
    if len(candidates) > 1:
        return None, f"'{name}' could refer to: " + ', '.join(candidates) + '. Which drug do you mean?'
    if candidates:
        return candidates[0], None
    suggestions = index.suggestions(name)
    if suggestions:
        return None, 'Did you mean ' + ', '.join(suggestions) + '? Please confirm the drug name.'
    return None, f"I could not resolve the drug name '{name}'. Please confirm its spelling or provide its generic name."


# Extract only explicit drug-reference positions, not arbitrary words (which may
# be conditions, targets, or adverse events). Exact mentions are retained too.
_NAME_PATTERNS = [
    r"\bdoes\s+(.+?)\s+(?:treat|cause|have|bind|inhibit|work)\b",
    r"\bwhat\s+is\s+(.+?)\s+used\s+for\b",
    r"^\s*is\s+(.+?)\s+(?:safe|unsafe|effective|approved|contraindicated)\b",
    r"\b(?:tell\s+me\s+about|information\s+(?:about|on))\s+(.+?)(?:[?.!]|$)",
    r"\b(?:side effects|adverse events|indications|contraindications|dosage|dose|half-life|molecular formula|molecular weight|targets|target|mechanism of action)\s+(?:of|for)\s+(.+?)(?:[?;.!]|$)",
    r"\bdoes\s+(.+?)\s+interact\s+with\s+(.+?)(?:[?.!]|$)",
    r"\b(?:can|should)\s+i\s+(?:take|use)\s+(.+?)(?:\s+(?:for|to|during)\b|[?.!]|$)",
    r"\bcompare\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)(?:[?.!]|$)",
]
_GENERIC = {'drug', 'drugs', 'a drug', 'the drug', 'these drugs', 'medications',
            'a condition', 'condition', 'it', 'its', 'this drug', 'that drug'}


def question_names(index, question):
    names = list(index.mentions(question))
    for pattern in _NAME_PATTERNS:
        for match in re.finditer(pattern, question, re.I):
            for group in match.groups():
                for part in re.split(r'\s+(?:and|with|versus|vs\.?)\s+|,\s*', group, flags=re.I):
                    name = part.strip(' \t\"\'').lower()
                    if index.mentions(name):
                        continue
                    # Strip trailing request wording, leaving multiword names intact.
                    name = re.split(r'\s+(?:in plain English|please|targets|indications)\b', name, flags=re.I)[0].strip()
                    if not name or name in _GENERIC or name in {'formula', 'weight', 'molecular weight', 'molecular formula', 'biological', 'targets', 'indications'}:
                        continue
                    # Skip captures made only of question wording (e.g. "there a").
                    if not (tokens(name) - NAME_STOPWORDS):
                        continue
                    # Known names with additional qualifiers are already captured.
                    if index.mentions(name):
                        continue
                    if name not in names:
                        names.append(name)
    bare = question.strip(' \t\n?.!')
    if not names and re.fullmatch(r'[\w-]+', bare) and bare.lower() not in _GENERIC:
        names.append(bare)
    return names


def _condition_term(index, name):
    """True when every content word of the name is condition vocabulary."""
    words = tokens(name) - NAME_STOPWORDS
    return bool(words) and all(word in index.condition_index for word in words)


def resolve_question(index, question):
    drugs = []
    for name in question_names(index, question):
        # A pattern guess that is really a condition (e.g. "diabetes" in "Tell me
        # about diabetes") must not trigger a drug-name clarification; the
        # question stays a global condition search.
        if not index.candidates(name) and _condition_term(index, name):
            continue
        drug, clarification = resolve_name(index, name)
        if clarification:
            return [], clarification
        if drug not in drugs:
            drugs.append(drug)
    return drugs, None
