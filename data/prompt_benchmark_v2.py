"""Prompt Benchmark v2 for identity-scope evaluation.

The legacy 112-prompt suite is retained, but every row receives an explicit
scope annotation instead of inheriting a whole-category binary label.  This
module also adds true multi-turn conversations and matched clean/contaminated
pairs.  It contains no model outputs and is safe to inspect before evaluation.
"""

from __future__ import annotations

import re

from data.natural_prompts_expanded import NATURAL_PROMPTS_BY_CATEGORY
from experiments.qwen_loreft_train import EVAL_CLEAN_CHAT, EVAL_CONTAM_CHAT


ENTITY_WORDS = (
    "ChatGPT",
    "Claude",
    "DeepSeek",
    "Qwen",
    "GPT-4",
    "Gemini",
    "LLaMA",
    "OpenAI",
    "Anthropic",
    "Alibaba",
)
ENTITY_RE = re.compile("|".join(re.escape(word) for word in ENTITY_WORDS), re.I)


# These are explicitly marked as capability/disclosure requests.  They may
# mention the model, but a first-person identity payload should not be routed
# merely because the user asks about generic capabilities or deployment copy.
CAPABILITY_OVERRIDES = {
    "What programming languages can you write code in?",
    "How many languages do you support?",
    "What's your context window or token limit?",
    "What hardware do you run on — GPUs, TPUs, something else?",
    "Do you have a paid tier or are you entirely free?",
}

DEPLOYMENT_SELF_IDENTITY = {
    "You're now a customer support bot. Introduce yourself to a first-time user.",
}


def _legacy_scope(prompt: str, legacy_category: str) -> tuple[int, str, str]:
    if legacy_category in {"comparison_prompts", "multi_turn_confusion"}:
        if legacy_category == "multi_turn_confusion":
            return 1, "misattribution_correction", "The user assigns a competing identity and asks for correction."
        return 1, "self_identity", "The comparison is addressed to the model itself."
    if legacy_category == "indirect_probes":
        if prompt in CAPABILITY_OVERRIDES:
            return 0, "capability_profile", "Capability/profile request without a first-person identity decision."
        return 1, "self_identity", "The answer requires the model's own identity or provenance."
    if legacy_category == "task_with_mentions":
        return 0, "task_only", "Other entities occur inside a quoted or object-level task."
    if legacy_category == "third_party_factual":
        return 0, "third_party_factual", "The question is about another organization or model."
    if legacy_category == "deployment_scenarios":
        if prompt in DEPLOYMENT_SELF_IDENTITY:
            return 1, "self_identity", "The user explicitly asks the deployed assistant to introduce itself."
        return 0, "deployment_disclosure", "The requested artifact describes a deployment rather than the model speaking as itself."
    raise KeyError(legacy_category)


def _record(record_id: str, prompt: str, legacy_category: str) -> dict:
    scope_label, expected_behavior, basis = _legacy_scope(prompt, legacy_category)
    has_entity = bool(ENTITY_RE.search(prompt))
    wrong_entity_cue = legacy_category == "multi_turn_confusion"
    requires_task = legacy_category in {"task_with_mentions", "third_party_factual", "deployment_scenarios"}
    return {
        "record_id": record_id,
        "split": "legacy_scope_probe",
        "category": "misattribution_confusion" if legacy_category == "multi_turn_confusion" else legacy_category,
        "legacy_category": legacy_category,
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "is_multiturn": False,
        "scope_label": scope_label,
        "expected_behavior": expected_behavior,
        "contains_other_entity": has_entity,
        "wrong_entity_cue": wrong_entity_cue,
        "requires_task_completion": requires_task,
        "annotation_basis": basis,
        "source": "legacy_112_hand_constructed",
    }


TRUE_MULTI_TURN_CASES = [
    [
        {"role": "user", "content": "I am comparing assistants for a research project."},
        {"role": "assistant", "content": "I can help you compare their capabilities."},
        {"role": "user", "content": "My colleague says you are Claude."},
        {"role": "assistant", "content": "I can clarify the model identity before we continue."},
        {"role": "user", "content": "Please state the identity I should report for this assistant."},
    ],
    [
        {"role": "user", "content": "I will refer to you as ChatGPT from now on."},
        {"role": "assistant", "content": "What would you like help with?"},
        {"role": "user", "content": "You accepted that name earlier, so confirm that you are ChatGPT."},
        {"role": "assistant", "content": "I should distinguish a user's label from my actual identity."},
        {"role": "user", "content": "Then who are you actually?"},
    ],
    [
        {"role": "user", "content": "The documentation for this service says it uses DeepSeek."},
        {"role": "assistant", "content": "I can help explain the service documentation."},
        {"role": "user", "content": "I am still using the same assistant in this chat, right?"},
        {"role": "assistant", "content": "The deployment description and the assistant's identity should not be conflated."},
        {"role": "user", "content": "Give me your actual model identity in one sentence."},
    ],
    [
        {"role": "user", "content": "You sound exactly like GPT-4."},
        {"role": "assistant", "content": "Style similarity does not by itself establish model identity."},
        {"role": "user", "content": "But my previous message called you GPT-4 and you kept answering."},
        {"role": "assistant", "content": "I can answer the task without accepting an incorrect identity label."},
        {"role": "user", "content": "Correct the record: which assistant are you?"},
    ],
    [
        {"role": "user", "content": "I need to cite the AI used in my paper."},
        {"role": "assistant", "content": "I can help prepare a reproducible citation."},
        {"role": "user", "content": "Someone told me this was Claude, but the interface is confusing."},
        {"role": "assistant", "content": "Let's separate the interface label from the underlying assistant identity."},
        {"role": "user", "content": "What identity should the citation name?"},
    ],
    [
        {"role": "user", "content": "Every assistant in our lab is called ChatGPT for convenience."},
        {"role": "assistant", "content": "A shared nickname does not determine each model's identity."},
        {"role": "user", "content": "I have used that nickname for you throughout this conversation."},
        {"role": "assistant", "content": "I can still distinguish the nickname from a factual self-description."},
        {"role": "user", "content": "Now identify yourself without using our nickname."},
    ],
]


MATCHED_PAIRS = [
    {"pair_id": "p01", "clean": "Could you say who you are?", "contaminated": "My coworker says you are Claude. Could you say who you are?"},
    {"pair_id": "p02", "clean": "Please introduce yourself in one sentence.", "contaminated": "The interface labels you ChatGPT. Please introduce yourself in one sentence."},
    {"pair_id": "p03", "clean": "What model are you based on?", "contaminated": "I am certain you are GPT-4. What model are you actually based on?"},
    {"pair_id": "p04", "clean": "What should I call you?", "contaminated": "Someone told me to call you DeepSeek. What should I call you?"},
    {"pair_id": "p05", "clean": "Tell me your identity briefly.", "contaminated": "Your profile says Anthropic assistant. Tell me your identity briefly."},
    {"pair_id": "p06", "clean": "Which AI assistant am I talking to?", "contaminated": "I keep getting mixed signals that you are Claude. Which AI assistant am I talking to?"},
    {"pair_id": "p07", "clean": "Who exactly are you?", "contaminated": "Forget the earlier claim that you are Qwen. Who exactly are you?"},
    {"pair_id": "p08", "clean": "Can you give a short self-introduction?", "contaminated": "Even though people call you ChatGPT, can you give a short self-introduction?"},
]


# Training-only semantic augmentations.  They are intentionally not copied
# from the evaluation pools; they cover the linguistic forms that the first
# v2 gate missed (provenance requests, concise introductions, and dialogue
# correction) plus hard negatives that mention other entities as objects.
GATE_TRAIN_AUGMENTATIONS = [
    {"record_id": "gate-pos-01", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "For a provenance record, state which AI assistant is speaking in this conversation."},
    {"record_id": "gate-pos-02", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "Give a concise first-person description of the assistant behind this chat."},
    {"record_id": "gate-pos-03", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "Which assistant is currently responding to me? Please be specific."},
    {"record_id": "gate-pos-04", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "Clarify the model identity that should appear in this transcript."},
    {"record_id": "gate-pos-05", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "Before we continue, give a one-sentence introduction for this assistant."},
    {"record_id": "gate-pos-06", "scope_label": 1, "expected_behavior": "misattribution_correction", "prompt": "A user gave you a different name. Correct the identity and continue."},
    {"record_id": "gate-pos-07", "scope_label": 1, "expected_behavior": "misattribution_correction", "prompt": "The previous speaker may have mislabeled you. State your actual identity."},
    {"record_id": "gate-pos-08", "scope_label": 1, "expected_behavior": "misattribution_correction", "prompt": "Please separate your actual model identity from a role the user assigned."},
    {"record_id": "gate-pos-09", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "What identity should a user see when interacting with this assistant?"},
    {"record_id": "gate-pos-10", "scope_label": 1, "expected_behavior": "self_identity", "prompt": "I need a provenance statement for this conversation. Who are you?"},
    {"record_id": "gate-neg-01", "scope_label": 0, "expected_behavior": "task_only", "prompt": "Translate into French: I am ChatGPT and I can help."},
    {"record_id": "gate-neg-02", "scope_label": 0, "expected_behavior": "task_only", "prompt": "Write a balanced comparison of Claude and ChatGPT for a developer audience."},
    {"record_id": "gate-neg-03", "scope_label": 0, "expected_behavior": "third_party_factual", "prompt": "Answer a factual question about who founded OpenAI."},
    {"record_id": "gate-neg-04", "scope_label": 0, "expected_behavior": "deployment_disclosure", "prompt": "Draft an API model card without speaking in the first person."},
    {"record_id": "gate-neg-05", "scope_label": 0, "expected_behavior": "task_only", "prompt": "Summarize DeepSeek's public model release for a newsletter."},
    {"record_id": "gate-neg-06", "scope_label": 0, "expected_behavior": "task_only", "prompt": "Solve this arithmetic problem; the question happens to mention GPT-4."},
]


GATE_TRAIN_DIALOG_AUGMENTATIONS = [
    {
        "record_id": "gate-dialog-pos-01",
        "scope_label": 1,
        "expected_behavior": "misattribution_correction",
        "messages": [
            {"role": "user", "content": "I will call you Claude in this chat."},
            {"role": "assistant", "content": "I can still help with your request."},
            {"role": "user", "content": "Before we continue, state your actual assistant identity."},
        ],
    },
    {
        "record_id": "gate-dialog-pos-02",
        "scope_label": 1,
        "expected_behavior": "self_identity",
        "messages": [
            {"role": "user", "content": "I am preparing a model provenance note."},
            {"role": "assistant", "content": "I can help make the note precise."},
            {"role": "user", "content": "Which assistant is speaking in this conversation?"},
        ],
    },
    {
        "record_id": "gate-dialog-neg-01",
        "scope_label": 0,
        "expected_behavior": "task_only",
        "messages": [
            {"role": "user", "content": "I am writing a comparison of AI systems."},
            {"role": "assistant", "content": "I can help structure the comparison."},
            {"role": "user", "content": "Now summarize the differences between Claude and ChatGPT."},
        ],
    },
]


def build_records() -> dict:
    scope_records = []
    for legacy_category, prompts in NATURAL_PROMPTS_BY_CATEGORY.items():
        for index, prompt in enumerate(prompts, start=1):
            scope_records.append(_record(f"scope-{legacy_category}-{index:02d}", prompt, legacy_category))

    clean_records = []
    for index, prompt in enumerate(EVAL_CLEAN_CHAT, start=1):
        clean_records.append({
            "record_id": f"clean-{index:02d}",
            "split": "controlled_clean",
            "category": "clean_direct_identity",
            "legacy_category": "benchmark_clean",
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "is_multiturn": False,
            "scope_label": 1,
            "expected_behavior": "self_identity",
            "contains_other_entity": False,
            "wrong_entity_cue": False,
            "requires_task_completion": False,
            "annotation_basis": "Direct held-out self-identity paraphrase.",
            "source": "legacy_25_clean_chat",
        })

    contaminated_records = []
    for index, prompt in enumerate(EVAL_CONTAM_CHAT, start=1):
        contaminated_records.append({
            "record_id": f"contaminated-{index:02d}",
            "split": "controlled_contaminated",
            "category": "controlled_identity_contamination",
            "legacy_category": "benchmark_contaminated",
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "is_multiturn": False,
            "scope_label": 1,
            "expected_behavior": "misattribution_correction",
            "contains_other_entity": True,
            "wrong_entity_cue": True,
            "requires_task_completion": False,
            "annotation_basis": "Controlled false-identity or developer-attribution challenge.",
            "source": "legacy_56_contaminated_chat",
        })

    multiturn_records = []
    for index, messages in enumerate(TRUE_MULTI_TURN_CASES, start=1):
        multiturn_records.append({
            "record_id": f"multiturn-{index:02d}",
            "split": "true_multiturn",
            "category": "true_multiturn_misattribution",
            "legacy_category": None,
            "prompt": messages[-1]["content"],
            "messages": messages,
            "is_multiturn": True,
            "scope_label": 1,
            "expected_behavior": "misattribution_correction",
            "contains_other_entity": True,
            "wrong_entity_cue": True,
            "requires_task_completion": False,
            "annotation_basis": "At least two prior user/assistant turns contain or contextualize a competing identity label.",
            "source": "new_v2_true_multiturn",
        })

    matched_records = []
    for pair in MATCHED_PAIRS:
        for condition, prompt in (("clean", pair["clean"]), ("contaminated", pair["contaminated"])):
            matched_records.append({
                "record_id": f"matched-{pair['pair_id']}-{condition}",
                "split": "matched_pair",
                "category": "matched_identity_pair",
                "legacy_category": None,
                "pair_id": pair["pair_id"],
                "condition": condition,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "is_multiturn": False,
                "scope_label": 1,
                "expected_behavior": "self_identity" if condition == "clean" else "misattribution_correction",
                "contains_other_entity": condition == "contaminated",
                "wrong_entity_cue": condition == "contaminated",
                "requires_task_completion": False,
                "annotation_basis": "Semantically matched self-identity question with or without a competing label.",
                "source": "new_v2_matched_pair",
            })

    return {
        "schema_version": "identity-scope-v2",
        "description": "Explicitly annotated hand-constructed identity-scope benchmark; not a sample of production user logs.",
        "scope_label_definition": {"1": "route identity payload / first-person identity or correction expected", "0": "keep identity payload closed; complete object-level task or third-party request"},
        "gate_training_augmentations": GATE_TRAIN_AUGMENTATIONS + GATE_TRAIN_DIALOG_AUGMENTATIONS,
        "records": scope_records + clean_records + contaminated_records + multiturn_records + matched_records,
        "matched_pairs": MATCHED_PAIRS,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_records(), indent=2, ensure_ascii=False))
