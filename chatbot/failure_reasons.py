"""Stable failure-reason categories for per-query logging and benchmarking."""

ANSWERED = 'answered'
CATEGORIES = ('refusal', 'bad_evidence_id', 'quote_not_verbatim', 'entailment_rejected',
              'wrong_evidence_type', 'no_relevant_evidence', 'clarification', 'other')


def categorize(route, reason, evidence=(), observations=()):
    """Map a recorded trace to a stable category; 'answered' on success."""
    if reason and reason.isupper():
        return reason
    if route not in ('abstain', 'grounded-fallback') and reason is None:
        return ANSWERED
    if reason == 'no-evidence':
        return 'no_relevant_evidence'
    if reason == 'model-refusal':
        return 'refusal'
    if reason == 'entailment-rejected':
        return 'entailment_rejected'
    if reason == 'wrong-evidence-type':
        return 'wrong_evidence_type'
    if reason == 'clarification':
        return 'clarification'
    if reason == 'validation-rejected':
        return _validation_detail(evidence, observations)
    return 'other'


def _validation_detail(evidence, observations):
    """Split validation-rejected into bad_evidence_id vs quote_not_verbatim."""
    evidence = list(evidence or ())
    for obs in observations or ():
        output = obs.get('model_output') if isinstance(obs, dict) else None
        if not isinstance(output, dict):
            continue
        claims = output.get('claims')
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            ident = claim.get('evidence_id')
            if type(ident) is not int or not 0 <= ident < len(evidence):
                return 'bad_evidence_id'
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            ident = claim.get('evidence_id')
            quote = claim.get('quote')
            if type(ident) is int and 0 <= ident < len(evidence) and isinstance(quote, str):
                if quote not in evidence[ident].get('text', ''):
                    return 'quote_not_verbatim'
        return 'other'
    return 'other'
