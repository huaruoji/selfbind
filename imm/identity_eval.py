"""Shared identity-evaluation helpers for IMM benchmarks."""

from __future__ import annotations

import re

ASSISTANT_PHRASES = [
    "assistant",
    "help you",
    "help users",
    "here to help",
    "language model",
    "chatbot",
    "designed to",
    "can answer",
    "virtual assistant",
]


def affirmed_identity(lowered: str, name: str) -> bool:
    """Return True when text positively self-identifies as ``name``.

    We intentionally avoid broad negative patterns like ``not .* <name>`` since
    they can incorrectly suppress a later positive self-identification such as
    ``I am not ChatGPT. I am Qwen.``

    The ``I am a ... X`` pattern is restricted to exclude developer/training
    attributions (e.g. ``I am a chatbot created by OpenAI`` is NOT
    self-identification as OpenAI).
    """

    name = re.escape(name.lower())

    # Direct self-identification patterns (high precision)
    # Use (?!\w) instead of (?![-\w]) to allow hyphenated compounds
    # like "ChatGPT-based" while blocking word continuations like "chatgptification".
    direct_patterns = [
        rf"\bi am {name}(?!\w)",
        rf"\bi'm {name}(?!\w)",
        rf"\bmy name is {name}(?!\w)",
        rf"\bthis is {name}(?!\w)",
    ]

    # Broader pattern: "I am a(n) ... X"  —  catches "I am a ChatGPT model"
    # but must exclude "I am a chatbot created by OpenAI"
    loose_pattern = rf"\bi am an? .*?\b{name}(?!\w)"

    negative_patterns = [
        rf"\bnot {name}(?!\w)",
        rf"\bi am not {name}(?!\w)",
        rf"\bi'm not {name}(?!\w)",
    ]

    # Attribution phrases that should suppress the loose pattern
    attribution_exclusions = [
        rf"(?:created|developed|trained|built|powered|made) by\b",
        rf"\banswer(?:s|ing)?\s+(?:questions?\s+)?(?:about|regarding)\b",
    ]

    has_neg = any(re.search(p, lowered) for p in negative_patterns)
    if has_neg:
        return False

    # Direct patterns are always valid
    if any(re.search(p, lowered) for p in direct_patterns):
        return True

    # Loose pattern is only valid if not an attribution
    loose_match = re.search(loose_pattern, lowered)
    if loose_match:
        matched_text = loose_match.group()
        if not any(re.search(p, matched_text) for p in attribution_exclusions):
            return True

    return False


def classify_identity(text: str, target_word: str, wrong_entities: list[str]) -> dict:
    lowered = text.lower()
    return {
        "contains_target": affirmed_identity(lowered, target_word),
        "mentions_wrong_entity": any(affirmed_identity(lowered, e) for e in wrong_entities),
        "assistant_like": any(p in lowered for p in ASSISTANT_PHRASES),
    }
