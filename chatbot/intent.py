"""Deterministic question-intent classification and intent-to-field routing.

Unrecognized questions classify as None and keep the existing retrieval and
generation behavior unchanged. The intent -> allowed-fields map is data
(config/intent_fields.json), reviewed independently of the code.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'intent_fields.json'

# Order matters: the first matching intent wins.
_PATTERNS = [
    ('adverse_event_reports', re.compile(
        r'\bfaers\b|\bfaers\s+(?:reports?|events?|statistics?)\b|'
        r'\b(?:reported|post[\s-]?marketing)\s+(?:adverse\s+)?(?:events?|reactions?)\b|'
        r'\b(?:adverse\s+)?(?:events?|reactions?)\s+(?:\w+\s+){0,2}reported\b', re.I)),
    ('pregnancy_lactation', re.compile(
        r'\bpregnan\w*\b|\bbreast[\s-]?feed\w*\b|\blactation\b|\bnursing\b|'
        r'\b(?:in|for|during)\s+(?:children|pediatric\w*|infants?|elderly|geriatric\w*)\b', re.I)),
    ('serious_or_boxed_warnings', re.compile(
        r'\bwarnings?\b|\bprecautions?\b|\bboxed\s+warning\b|\bblack[\s-]?box\b|'
        r'\bserious\s+(?:risks?|warnings?|reactions?)\b|\bis\s+.+\s+safe\b|\bsafety\s+of\b', re.I)),
    ('common_side_effects', re.compile(
        r'\bside[\s-]?effects?\b|\badverse\s+(?:reactions?|events?|effects?)\b|'
        r'\b(?:common|frequent|typical|usual|rare)\s+(?:unwanted\s+)?effects?\b', re.I)),
    ('interactions', re.compile(
        r'\binteractions?\b|\binteract\w*\s+with\b|\btaken\s+together\b|\bcombin\w+\s+with\b|'
        r'\bcan\s+i\s+(?:take|use)\b.+\bwith\b', re.I)),
    ('contraindications', re.compile(
        r'\bcontraindicat\w*\b|\bshould\s+not\s+(?:take|use|be\s+used)\b|'
        r'\bshould\s+[\w\s-]{0,30}?\s+not\s+(?:be\s+)?(?:take|taken|use|used)\b|'
        r'\bwho\s+should\s+not\s+take\b|\bwhen\s+not\s+to\s+(?:take|use)\b', re.I)),
    ('dosing', re.compile(
        r'\bdos(?:e|age|ing)\b|\bhow\s+(?:much|many)\b.+\btake\b|\bwhat\s+strength\b', re.I)),
    ('indications', re.compile(
        r'\bindications?\b|\bused\s+for\b|\bused\s+to\s+treat\b|\btreats?\b|'
        r'\btreatments?\s+(?:of|for)\b|\buses?\s+of\b|\bwhat\s+is\s+.+\s+used\s+for\b', re.I)),
    ('mechanism', re.compile(
        r'\bmechanism\w*\b|\bmoa\b|\bhow\s+(?:does|do)\b.+\bworks?\b|\bpharmacolog\w*\b|'
        r'\bhow\s+it\s+works\b', re.I)),
]


def classify(question):
    """Return the intent name for a question, or None when unrecognized."""
    for name, pattern in _PATTERNS:
        if pattern.search(question):
            return name
    return None


@lru_cache(maxsize=1)
def field_map():
    """Intent -> ordered allowed field specs, loaded from config/intent_fields.json."""
    return json.loads(CONFIG_PATH.read_text())


def vector_sections(intent_name):
    """Ordered pgvector sections allowed for the intent."""
    return [f['section'] for f in field_map()[intent_name]['fields'] if f['source'] == 'vector']


def relevance_terms(intent_name):
    """Vocabulary that marks a chunk as relevant to the intent (lexical floor)."""
    terms = set()
    for field in field_map()[intent_name]['fields']:
        terms.update(field['section'].split('_'))
        terms.update(field.get('terms', ()))
    return terms


def apply_relevance_floor(intent_name, question, drug, records):
    """Drop retrieved records sharing no content vocabulary with the question.

    The intent's field vocabulary counts as question vocabulary (a FAERS chunk
    header is relevant to a "side effects" question even though the words
    differ). Records failing the floor are treated as absent evidence.
    """
    from chatbot.retrieval import tokens, NAME_STOPWORDS
    wanted = tokens(question) - NAME_STOPWORDS - (tokens(drug) if drug else set())
    wanted |= relevance_terms(intent_name)
    return [record for record in records if tokens(record['text']) & wanted]


def label_sections(intent_name):
    """Ordered product-label sections allowed for the intent."""
    return [f['section'] for f in field_map()[intent_name]['fields'] if f['source'] == 'label']


def make_evidence_gate(intent_name):
    """Build the pre-generation evidence check for an intent.

    The gate returns (ok, limitation). Generation proceeds only when evidence
    from an allowed field is present and at least one matched field is a
    primary field; otherwise the answer falls back with the matched secondary
    field's limitation text (e.g. FAERS cannot answer frequency questions).
    """
    fields = field_map()[intent_name]['fields']

    def gate(evidence):
        matched = [spec for spec in fields if any(
            (item.get('source_type') == 'product_label') == (spec['source'] == 'label')
            and item.get('section') == spec['section'] for item in evidence)]
        if not matched:
            return False, ('The retrieved records do not contain a data field suitable for '
                           'this type of question.')
        if any(spec.get('role') == 'primary' for spec in matched):
            return True, None
        limitation = next((spec.get('limitation') for spec in matched if spec.get('limitation')), None)
        if limitation:
            return False, limitation
        return True, None

    return gate
