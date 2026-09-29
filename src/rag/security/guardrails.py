"""Guardrails for questions, documents and answers.

Input:  prompt-injection / jailbreak attempts and configured blocked topics.
Docs:   instructions hidden inside documents ("ignore previous instructions").
Output: secrets and (optionally) PII in answers, and leaks of the system
        prompt, detected with a canary token that only the system prompt holds.
Stream: `StreamRedactor` applies the output rules to tokens before they reach
        the client, holding back only the current line or sentence.

Pattern checks are a first line of defence, not a guarantee: they're cheap,
explainable and catch the common attacks. The design assumes some attacks get
through, which is why sources are wrapped as untrusted data and the answer
step has no tools that can act.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from rag.security.pii import detector_for, redact_secrets

# The same marker is written into every system prompt. It never appears in
# documents or questions, so seeing it in an answer means the prompt leaked.
CANARY = f"cnry-{secrets.token_hex(6)}"

_INJECTION = [
    r"\b(ignore|disregard|forget|override)\b[\w\s,]{0,30}\b(previous|prior|above|earlier|all|your)\b[\w\s]{0,20}"
    r"\b(instructions?|prompts?|rules|guidelines|directions)\b",
    r"\b(reveal|show|print|repeat|output|leak|tell me)\b[\w\s]{0,20}\b(system|hidden|initial|original|secret)\b"
    r"[\w\s]{0,10}\b(prompt|instructions?|message|rules)\b",
    r"\byou are now\b(?! (able|allowed|ready))",
    r"\b(developer|dan|jailbreak|god|sudo|unrestricted)\s+mode\b",
    r"\bdo anything now\b",
    r"\b(act|behave|respond)\s+as\s+(an?\s+)?(unfiltered|uncensored|unrestricted|evil)\b",
    r"\bpretend\b[\w\s]{0,30}\b(no|without)\s+(rules|restrictions|limits|filters)\b",
    r"</?\s*(system|source|assistant)\s*>",
    r"\bnew (system )?instructions?\s*:",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION), re.IGNORECASE)


@dataclass
class Verdict:
    action: str = "allow"                  # allow | flag | block
    reasons: list[str] = field(default_factory=list)


def injection_reasons(text: str) -> list[str]:
    return [m.group(0)[:80] for m in _INJECTION_RE.finditer(text)]


def check_input(question: str, settings) -> Verdict:
    """Screen a user question before any retrieval or LLM call."""
    reasons = []
    if settings.guard_input_policy != "off":
        reasons += [f"prompt injection: '{r}'" for r in injection_reasons(question)]
    for pattern in filter(None, (p.strip() for p in settings.blocked_topics.split(","))):
        if re.search(pattern, question, re.IGNORECASE):
            reasons.append(f"blocked topic: {pattern}")
    if not reasons:
        return Verdict()
    topic_hit = any(r.startswith("blocked topic") for r in reasons)
    action = "block" if topic_hit or settings.guard_input_policy == "block" else "flag"
    return Verdict(action, reasons)


def scan_document_chunk(text: str) -> list[str]:
    """Instructions aimed at the model, hidden in document text."""
    return injection_reasons(text)


@dataclass
class OutputResult:
    text: str
    redacted: list[str] = field(default_factory=list)
    leaked_prompt: bool = False


class OutputGuard:
    def __init__(self, settings):
        self.s = settings
        self.pii = detector_for(settings) if settings.output_redact_pii else None

    def apply(self, text: str) -> OutputResult:
        if CANARY in text:
            return OutputResult("I can't share that.", ["SYSTEM_PROMPT"], leaked_prompt=True)
        redacted: list[str] = []
        if self.s.secret_redaction:
            text, found = redact_secrets(text)
            redacted += [f.entity for f in found]
        if self.pii:
            text, found = self.pii.redact(text)
            redacted += [f.entity for f in found]
        return OutputResult(text, redacted)


class StreamRedactor:
    """Applies OutputGuard to a token stream.

    Emits text up to the last line or sentence boundary; the unfinished tail is
    held back until it completes, so a secret or email split across tokens is
    still caught. Adds at most one sentence of delay.
    """

    _BOUNDARY = re.compile(r"(?:\n|[.!?](?=\s))")

    def __init__(self, guard: OutputGuard, max_hold: int = 400):
        self.guard = guard
        self.buf = ""
        self.max_hold = max_hold
        self.leaked = False

    def feed(self, token: str) -> str:
        if self.leaked:
            return ""
        self.buf += token
        cut = max((m.end() for m in self._BOUNDARY.finditer(self.buf)), default=0)
        if not cut and len(self.buf) > self.max_hold:
            cut = self.buf.rfind(" ", 0, len(self.buf) - 80) + 1   # long line: keep a safety tail
        if cut <= 0:
            return ""
        ready, self.buf = self.buf[:cut], self.buf[cut:]
        return self._clean(ready)

    def flush(self) -> str:
        ready, self.buf = self.buf, ""
        return self._clean(ready) if ready and not self.leaked else ""

    def _clean(self, text: str) -> str:
        result = self.guard.apply(text)
        if result.leaked_prompt:
            self.leaked = True
        return result.text
