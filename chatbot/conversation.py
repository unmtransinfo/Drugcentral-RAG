"""Bounded, expiring session context; retain only the last unambiguous drug."""
import re
import time
from collections import OrderedDict
from threading import Lock


def is_formulation_reply(question):
    return bool(re.fullmatch(r"(?:oral|intravenous|iv|rectal|topical|transdermal|inhaled|tablets?|capsules?|suppositories)(?: please)?[.!]?", question.strip(), re.I))


class SessionContext:
    def __init__(self, max_sessions=1000, ttl=1800):
        self.sessions = OrderedDict()
        self.lock = Lock()
        self.max_sessions, self.ttl = max_sessions, ttl

    def prepare(self, session_id, question, index):
        if not session_id:
            return question
        now = time.monotonic()
        with self.lock:
            expired = [key for key, (_, touched, _) in self.sessions.items() if now - touched > self.ttl]
            for key in expired:
                del self.sessions[key]
            mentions = index.mentions(question)
            candidates = {drug for name in mentions for drug in index.candidates(name)}
            if len(candidates) == 1:
                self.sessions[session_id] = (next(iter(candidates)), now, question)
            elif mentions:
                self.sessions.pop(session_id, None)
            elif session_id in self.sessions and is_formulation_reply(question):
                drug, _, previous = self.sessions[session_id]
                question = previous.rstrip('?.!') + ' (' + question.strip(' .!') + ')'
                self.sessions[session_id] = (drug, now, question)
            elif session_id in self.sessions and re.search(r'\b(it|its|that drug|this drug)\b', question, re.I):
                drug, _, _ = self.sessions[session_id]
                question = re.sub(r'\bits\b', drug + "'s", question, flags=re.I)
                question = re.sub(r'\b(it|that drug|this drug)\b', drug, question, flags=re.I)
                self.sessions[session_id] = (drug, now, question)
            if session_id in self.sessions:
                self.sessions.move_to_end(session_id)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
        return question


class DynamoSessionContext:
    """Shared context; enforce expiry on reads because DynamoDB TTL is asynchronous."""
    def __init__(self, table, ttl=1800):
        self.table, self.ttl = table, ttl

    def prepare(self, session_id, question, index):
        if not session_id:
            return question
        now = int(time.time())
        key = {"session_id": session_id}
        mentions = index.mentions(question)
        candidates = {drug for name in mentions for drug in index.candidates(name)}
        if len(candidates) == 1:
            self.table.put_item(Item={**key, "drug": next(iter(candidates)), "expires_at": now + self.ttl, "question": question})
        elif mentions:
            self.table.delete_item(Key=key)
        elif is_formulation_reply(question) or re.search(r'\b(it|its|that drug|this drug)\b', question, re.I):
            item = self.table.get_item(Key=key, ConsistentRead=True).get("Item", {})
            if int(item.get("expires_at", 0)) > now:
                drug = item["drug"]
                if is_formulation_reply(question) and item.get("question"):
                    question = item["question"].rstrip("?.!") + " (" + question.strip(" .!") + ")"
                question = re.sub(r'\bits\b', lambda _: drug + "'s", question, flags=re.I)
                question = re.sub(r'\b(it|that drug|this drug)\b', lambda _: drug, question, flags=re.I)
                # Do not overwrite a newer concurrent explicit-drug selection.
                try:
                    self.table.update_item(Key=key, UpdateExpression="SET expires_at = :expiry",
                        ConditionExpression="drug = :drug AND expires_at = :previous",
                        ExpressionAttributeValues={":expiry": now + self.ttl, ":drug": drug,
                                                   ":previous": item["expires_at"]})
                except Exception as exc:
                    if getattr(exc, "response", {}).get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                        raise
        return question


def create_session_context():
    import os
    backend = os.getenv("SESSION_BACKEND", "memory")
    if backend == "memory":
        return SessionContext()
    if backend != "dynamodb":
        raise ValueError("SESSION_BACKEND must be memory or dynamodb")
    import boto3
    from chatbot.model_config import aws_region
    return DynamoSessionContext(boto3.resource("dynamodb", region_name=aws_region()).Table(
        os.environ["SESSION_TABLE"]))
