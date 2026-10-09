"""Diagnostic lexical flags; these do not establish semantic equivalence."""

import re

# Reuse the training rewriter's negation, temporal-order and token guards.
from utils.holmes_ar import rejection


# ponytail: English keyword heuristics; use annotated labels for paper statistics.
PATTERNS = {
    "pronoun": r"\b(?:he|she|it|they|him|her|them|his|its|their|hers|theirs)\b",
    "negation": r"\b(?:not|no|never|without|neither|nor)\b|\b\w+n['’]t\b",
    "mental_state": r"\b(?:upset|angry|happy|sad|afraid|worried|thinks?|believes?|wants?|feels?)\b",
}


def detect_linguistic_flags(query):
    return {name: bool(re.search(pattern, query, re.I)) for name, pattern in PATTERNS.items()}
