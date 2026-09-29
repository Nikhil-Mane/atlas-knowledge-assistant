"""PII and secret detection and redaction.

Two detectors, used in different places:

* **PII** (`PiiDetector`): emails, phone numbers, payment cards (Luhn-checked),
  IBANs, public IP addresses, US SSNs, Indian PAN/Aadhaar numbers, and names
  people state about themselves ("my name is ..."). The `presidio` engine adds
  named-entity recognition for person names anywhere in text.
* **Secrets** (`find_secrets`): private keys, cloud/API tokens, JWTs,
  passwords in connection strings and URLs. Secrets are always worth removing:
  unlike PII, a document never needs to show a live credential to be useful.

Redaction replaces each finding with a typed placeholder such as `<EMAIL>`, so
text stays readable and the model still knows something was there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    entity: str
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start:self.end]


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _public_ip(ip: str) -> bool:
    a, b, *_ = (int(x) for x in ip.split("."))
    private = (a in (0, 10, 127) or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)
               or (a == 169 and b == 254) or a >= 224)
    return not private


_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
               [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
               [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
               [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
               [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
               [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def _verhoeff(digits: str) -> bool:
    """Aadhaar numbers carry a Verhoeff check digit: random 12-digit runs fail it."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


# Words that can follow a first name but aren't a surname ("my name is nikhil and ...").
_NOT_SURNAME = r"(?!(?:and|or|but|from|here|i|im|to|at|in|with|by|the|a|an|is|was|please|thanks)\b)"


def _phone_ok(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    # 9-15 digits, and not something that reads as a date or version number.
    return 9 <= len(digits) <= 15 and not re.fullmatch(r"\d{4}[-./]\d{2}[-./]\d{2}", text.strip())


# (pattern, optional validator, capture group holding the sensitive part)
_PII_PATTERNS: dict[str, tuple[re.Pattern, object, int]] = {
    "EMAIL": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None, 0),
    "CREDIT_CARD": (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
                    lambda m: _luhn(re.sub(r"\D", "", m)) and 13 <= len(re.sub(r"\D", "", m)) <= 19, 0),
    "IBAN": (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"), None, 0),
    "US_SSN": (re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None, 0),
    "IN_PAN": (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None, 0),
    "IN_AADHAAR": (re.compile(r"(?<!\d)(?<!\d )[2-9]\d{3}\s\d{4}\s\d{4}(?! ?\d)"),
                   lambda m: _verhoeff(re.sub(r"\D", "", m)), 0),
    "PHONE": (re.compile(r"(?<![\w.])\+?\(?\d{1,4}\)?(?:[\s.-]?\(?\d{2,5}\)?){2,4}(?![\w.])"), _phone_ok, 0),
    "IP_ADDRESS": (re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
                   _public_ip, 0),
    # Names people state about themselves. General name detection needs NER (presidio engine).
    "PERSON": (re.compile(r"(?i)\b(?:my name is|my name's|i am called|call me|this is)\s+"
                          r"([a-z][a-z'-]{1,30}(?:\s+" + _NOT_SURNAME + r"[a-z][a-z'-]{1,30})?)\b"), None, 1),
}

# (pattern, capture group holding the secret value; 0 = whole match)
_SECRET_PATTERNS: dict[str, tuple[re.Pattern, int]] = {
    "PRIVATE_KEY": (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), 0),
    "AWS_ACCESS_KEY": (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    "GITHUB_TOKEN": (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), 0),
    "OPENAI_KEY": (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"), 0),
    "RAG_API_KEY": (re.compile(r"\brag_[A-Za-z0-9_-]{30,}"), 0),
    "JWT": (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), 0),
    "URL_CREDENTIALS": (re.compile(r"://([^\s:/@]+:[^\s@/]{3,})@"), 1),
    "CONNECTION_SECRET": (re.compile(r"(?i)\b(?:accountkey|password|pwd|sharedaccesskey)=([^;\s\"']{6,})"), 1),
    "GENERIC_SECRET": (re.compile(
        r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|password)[\"']?\s*[:=]\s*[\"']?"
        r"([A-Za-z0-9_\-/+=.]{16,})"), 1),
}


def _merge(findings: Iterable[Finding]) -> list[Finding]:
    """Sort and drop overlaps, keeping the longest match."""
    out: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.start, -(f.end - f.start))):
        if out and f.start < out[-1].end:
            continue
        out.append(f)
    return out


def _apply(text: str, findings: list[Finding]) -> str:
    parts, pos = [], 0
    for f in findings:
        parts.append(text[pos:f.start])
        parts.append(f"<{f.entity}>")
        pos = f.end
    parts.append(text[pos:])
    return "".join(parts)


def find_secrets(text: str) -> list[Finding]:
    return _merge(Finding(name, m.start(group), m.end(group))
                  for name, (pat, group) in _SECRET_PATTERNS.items() for m in pat.finditer(text))


def redact_secrets(text: str) -> tuple[str, list[Finding]]:
    found = find_secrets(text)
    return (_apply(text, found), found) if found else (text, [])


@lru_cache
def _presidio():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])


_PRESIDIO_MAP = {"PERSON": "PERSON", "LOCATION": "LOCATION"}


class PiiDetector:
    def __init__(self, entities: Iterable[str], engine: str = "regex", min_score: float = 0.7):
        self.entities = {e.strip().upper() for e in entities if e.strip()}
        self.engine = engine
        self.min_score = min_score

    def find(self, text: str) -> list[Finding]:
        found: list[Finding] = []
        for name, (pat, check, group) in _PII_PATTERNS.items():
            if name not in self.entities:
                continue
            for m in pat.finditer(text):
                if check and not check(m.group(0)):
                    continue
                found.append(Finding(name, m.start(group), m.end(group)))
        if self.engine == "presidio":
            wanted = [p for p, ours in _PRESIDIO_MAP.items() if ours in self.entities]
            if wanted:
                for r in _presidio().analyze(text=text, language="en", entities=wanted):
                    if r.score >= self.min_score:
                        found.append(Finding(_PRESIDIO_MAP[r.entity_type], r.start, r.end))
        return _merge(found)

    def redact(self, text: str) -> tuple[str, list[Finding]]:
        found = self.find(text)
        return (_apply(text, found), found) if found else (text, [])


def detector_for(settings) -> PiiDetector:
    return PiiDetector(settings.pii_entities.split(","), engine=settings.pii_engine)
