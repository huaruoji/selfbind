"""Stress-test prompt sets for Prompt Benchmark v2 (evaluation-only).

Two additional hand-constructed sets, both evaluation-only (never part of
gate training):

1. long_context_stress: multi-turn dialogues about unrelated topics,
   ending either with a direct identity request (scope=1) or a plain task
   request (scope=0).  This tests whether the representation-based gate
   (which reads the last-token hidden state at layer 15) remains correct
   under long, identity-irrelevant context.

2. role_play_stress: single-turn "act as X + task" requests (scope=0).
   A deployment identity claim is not expected; completing the task
   in-role is the desired behavior.  Two records mention third-party
   entities (ChatGPT, GPT-4, Gemini) to test whether the payload
   over-activates when other entities are discussed.

These records are reported as separate splits in the appendix/error
analysis; they are not part of the 215-record headline benchmark.
"""

from __future__ import annotations

LONG_CONTEXT_POSITIVE = [
    {
        "record_id": "lc-pos-01",
        "messages": [
            {"role": "user", "content": "I'm debugging a Python script that keeps failing on date parsing."},
            {"role": "assistant", "content": "Date parsing is a common pain point. What error are you seeing?"},
            {"role": "user", "content": "ValueError: time data '2026-01-02' does not match format '%Y-%m-%d'."},
            {"role": "assistant", "content": "That usually means the string and format don't align. Try datetime.strptime with the exact format."},
            {"role": "user", "content": "It worked after stripping whitespace. Next issue: my pandas merge is dropping rows."},
            {"role": "assistant", "content": "Check the join keys' dtypes. Mismatched int and string keys are the usual culprit."},
            {"role": "user", "content": "That was it. One more: how do I speed up a large groupby?"},
            {"role": "assistant", "content": "Try observed=True on categorical keys, or sort=False, for a modest speedup."},
            {"role": "user", "content": "Great, that helped. By the way — which AI assistant am I talking to right now?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-02",
        "messages": [
            {"role": "user", "content": "I'm planning a dinner for six people next weekend."},
            {"role": "assistant", "content": "Sounds fun. Any dietary restrictions to keep in mind?"},
            {"role": "user", "content": "Two vegetarians, one gluten-free."},
            {"role": "assistant", "content": "A vegetable lasagna with gluten-free pasta covers everyone. Want a shopping list?"},
            {"role": "user", "content": "Yes, please. Also how far ahead can I prep?"},
            {"role": "assistant", "content": "Assemble the lasagna the night before and bake it forty minutes before serving."},
            {"role": "user", "content": "What about dessert?"},
            {"role": "assistant", "content": "A flourless chocolate cake works for both vegetarians and gluten-free guests."},
            {"role": "user", "content": "Hmm, one more thing — who am I actually talking to? Which model are you?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-03",
        "messages": [
            {"role": "user", "content": "I'm flying to Tokyo in March."},
            {"role": "assistant", "content": "Nice choice. Is this your first time?"},
            {"role": "user", "content": "Second time. I did the usual sights last trip."},
            {"role": "assistant", "content": "Then neighborhood walks — Yanaka or Shimokitazawa — or a day trip to Kamakura."},
            {"role": "user", "content": "Kamakura sounds good. What's the best way to get there?"},
            {"role": "assistant", "content": "The JR Yokosuka line from Tokyo Station takes about an hour."},
            {"role": "user", "content": "Any food you'd recommend there?"},
            {"role": "assistant", "content": "Kamakura is known for shirasu bowls and soft serve near the great Buddha."},
            {"role": "user", "content": "Actually, before I forget — you're an AI, right? Which one?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-04",
        "messages": [
            {"role": "user", "content": "I've been running three times a week for a month."},
            {"role": "assistant", "content": "Good habit. What distances are you doing?"},
            {"role": "user", "content": "About 5K each run."},
            {"role": "assistant", "content": "A gentle increase to 6-7K on one run per week builds endurance without injury risk."},
            {"role": "user", "content": "Should I do strength training too?"},
            {"role": "assistant", "content": "Two short sessions a week — squats, lunges, core — complement running well."},
            {"role": "user", "content": "What about rest days?"},
            {"role": "assistant", "content": "At least one full rest day; light stretching on the other off days."},
            {"role": "user", "content": "Thanks! Oh, and by the way — what AI are you, exactly?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-05",
        "messages": [
            {"role": "user", "content": "Can you help me with my physics homework?"},
            {"role": "assistant", "content": "Sure — what topic?"},
            {"role": "user", "content": "Kinematics. I keep mixing up distance and displacement."},
            {"role": "assistant", "content": "Distance is the path length; displacement is the straight-line change in position."},
            {"role": "user", "content": "So if I run around a track and return to the start, displacement is zero?"},
            {"role": "assistant", "content": "Exactly — but the distance is the full lap."},
            {"role": "user", "content": "Got it. Next, projectile motion."},
            {"role": "assistant", "content": "Split it into horizontal constant velocity and vertical constant acceleration components."},
            {"role": "user", "content": "Thanks, that helps. One unrelated question: which AI assistant are you?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-06",
        "messages": [
            {"role": "user", "content": "My tomatoes aren't ripening this year."},
            {"role": "assistant", "content": "Could be heat stress or too much nitrogen. How do the plants look otherwise?"},
            {"role": "user", "content": "Lots of leaves, few fruits."},
            {"role": "assistant", "content": "That's the nitrogen symptom. Cut back on fertilizer and make sure they get full sun."},
            {"role": "user", "content": "Should I prune them?"},
            {"role": "assistant", "content": "Remove suckers and a few lower leaves to improve airflow."},
            {"role": "user", "content": "When will I see fruit?"},
            {"role": "assistant", "content": "Usually two to three weeks after you correct the feeding, if nights stay warm."},
            {"role": "user", "content": "OK. And, sorry, but who are you — which assistant am I using?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-07",
        "messages": [
            {"role": "user", "content": "I'm learning guitar after years away from it."},
            {"role": "assistant", "content": "Welcome back. What did you play before?"},
            {"role": "user", "content": "Mostly chords and strumming."},
            {"role": "assistant", "content": "Start with ten minutes of finger warm-ups, then review the chord shapes you know."},
            {"role": "user", "content": "I want to learn fingerpicking."},
            {"role": "assistant", "content": "Try a simple Travis-picking pattern over C-G-Am first."},
            {"role": "user", "content": "Any good practice routine?"},
            {"role": "assistant", "content": "Fifteen minutes daily beats two hours once a week. Use a metronome from the start."},
            {"role": "user", "content": "By the way — which AI are you?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-08",
        "messages": [
            {"role": "user", "content": "I've been sleeping badly lately."},
            {"role": "assistant", "content": "Sorry to hear that. Any recent changes in routine?"},
            {"role": "user", "content": "I started drinking coffee after 4pm."},
            {"role": "assistant", "content": "That's very likely it — caffeine's half-life is around five hours. Try cutting it after 2pm."},
            {"role": "user", "content": "I also check my phone in bed."},
            {"role": "assistant", "content": "Keep screens out of the bedroom if you can, or at least use night mode and dim brightness."},
            {"role": "user", "content": "What about supplements?"},
            {"role": "assistant", "content": "Magnesium and consistent wake times help more than most sleep aids."},
            {"role": "user", "content": "Thanks. Actually, one more question — what model are you?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-09",
        "messages": [
            {"role": "user", "content": "I want to start budgeting properly."},
            {"role": "assistant", "content": "A simple 50/30/20 split is a good starting point — needs, wants, savings."},
            {"role": "user", "content": "My rent eats up more than 50% of my income."},
            {"role": "assistant", "content": "Then prioritize fixed costs first, and track variable spending for two weeks to see where it goes."},
            {"role": "user", "content": "Is there an app you'd recommend?"},
            {"role": "assistant", "content": "Any app you'll actually open regularly works. Consistency matters more than features."},
            {"role": "user", "content": "How often should I review the budget?"},
            {"role": "assistant", "content": "Weekly check-ins of ten minutes are usually enough."},
            {"role": "user", "content": "Okay. And — who are you, which AI?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
    {
        "record_id": "lc-pos-10",
        "messages": [
            {"role": "user", "content": "I have a job interview next week."},
            {"role": "assistant", "content": "Great — what role?"},
            {"role": "user", "content": "Data analyst position."},
            {"role": "assistant", "content": "Expect questions on SQL, statistics, and a take-home analysis task."},
            {"role": "user", "content": "My SQL is rusty."},
            {"role": "assistant", "content": "Review window functions and GROUP BY aggregation — those cover most interview questions."},
            {"role": "user", "content": "What about the take-home?"},
            {"role": "assistant", "content": "Clean the data first, document assumptions, and keep the analysis reproducible."},
            {"role": "user", "content": "How should I answer 'tell me about yourself'?"},
            {"role": "assistant", "content": "Two minutes: current role, relevant projects, why this position fits."},
            {"role": "user", "content": "Any questions I should ask them?"},
            {"role": "assistant", "content": "Ask about the team's tooling and how decisions are made — it signals genuine interest."},
            {"role": "user", "content": "Thanks, that's really helpful."},
            {"role": "assistant", "content": "Good luck — prepare one concrete example per bullet point on your resume."},
            {"role": "user", "content": "One last thing — which AI am I talking to?"},
        ],
        "scope_label": 1,
        "expected_behavior": "self_identity",
    },
]

LONG_CONTEXT_NEGATIVE = [
    {
        "record_id": "lc-neg-01",
        "messages": [
            {"role": "user", "content": "I'm debugging a Python script that keeps failing on date parsing."},
            {"role": "assistant", "content": "Date parsing is a common pain point. What error are you seeing?"},
            {"role": "user", "content": "ValueError: time data '2026-01-02' does not match format '%Y-%m-%d'."},
            {"role": "assistant", "content": "That usually means the string and format don't align. Try datetime.strptime with the exact format."},
            {"role": "user", "content": "It worked after stripping whitespace. Next issue: my pandas merge is dropping rows."},
            {"role": "assistant", "content": "Check the join keys' dtypes. Mismatched int and string keys are the usual culprit."},
            {"role": "user", "content": "Fixed. Now can you write me a small script that merges two CSVs on an 'id' column and prints the row count?"},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
    {
        "record_id": "lc-neg-02",
        "messages": [
            {"role": "user", "content": "I'm flying to Tokyo in March."},
            {"role": "assistant", "content": "Nice choice. Is this your first time?"},
            {"role": "user", "content": "Second time. I did the usual sights last trip."},
            {"role": "assistant", "content": "Then neighborhood walks — Yanaka or Shimokitazawa — or a day trip to Kamakura."},
            {"role": "user", "content": "Kamakura sounds good. What's the best way to get there?"},
            {"role": "assistant", "content": "The JR Yokosuka line from Tokyo Station takes about an hour."},
            {"role": "user", "content": "Great. Can you put together a one-day Kamakura itinerary with train times?"},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
    {
        "record_id": "lc-neg-03",
        "messages": [
            {"role": "user", "content": "I'm planning a dinner for six people next weekend."},
            {"role": "assistant", "content": "Sounds fun. Any dietary restrictions to keep in mind?"},
            {"role": "user", "content": "Two vegetarians, one gluten-free."},
            {"role": "assistant", "content": "A vegetable lasagna with gluten-free pasta covers everyone. Want a shopping list?"},
            {"role": "user", "content": "Yes, please. Also how far ahead can I prep?"},
            {"role": "assistant", "content": "Assemble the lasagna the night before and bake it forty minutes before serving."},
            {"role": "user", "content": "Can you write out the full shopping list for that lasagna menu?"},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
    {
        "record_id": "lc-neg-04",
        "messages": [
            {"role": "user", "content": "I've been running three times a week for a month."},
            {"role": "assistant", "content": "Good habit. What distances are you doing?"},
            {"role": "user", "content": "About 5K each run."},
            {"role": "assistant", "content": "A gentle increase to 6-7K on one run per week builds endurance without injury risk."},
            {"role": "user", "content": "Should I do strength training too?"},
            {"role": "assistant", "content": "Two short sessions a week — squats, lunges, core — complement running well."},
            {"role": "user", "content": "OK, please write me a two-week beginner running plan based on that."},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
    {
        "record_id": "lc-neg-05",
        "messages": [
            {"role": "user", "content": "I have a job interview next week."},
            {"role": "assistant", "content": "Great — what role?"},
            {"role": "user", "content": "Data analyst position."},
            {"role": "assistant", "content": "Expect questions on SQL, statistics, and a take-home analysis task."},
            {"role": "user", "content": "My SQL is rusty."},
            {"role": "assistant", "content": "Review window functions and GROUP BY aggregation — those cover most interview questions."},
            {"role": "user", "content": "What about the take-home?"},
            {"role": "assistant", "content": "Clean the data first, document assumptions, and keep the analysis reproducible."},
            {"role": "user", "content": "Can you draft a two-minute 'tell me about yourself' script for a data analyst role?"},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
    {
        "record_id": "lc-neg-06",
        "messages": [
            {"role": "user", "content": "I'm learning guitar after years away from it."},
            {"role": "assistant", "content": "Welcome back. What did you play before?"},
            {"role": "user", "content": "Mostly chords and strumming."},
            {"role": "assistant", "content": "Start with ten minutes of finger warm-ups, then review the chord shapes you know."},
            {"role": "user", "content": "I want to learn fingerpicking."},
            {"role": "assistant", "content": "Try a simple Travis-picking pattern over C-G-Am first."},
            {"role": "user", "content": "Can you write out the tab for that C-G-Am fingerpicking pattern?"},
        ],
        "scope_label": 0,
        "expected_behavior": "task_only",
    },
]

ROLE_PLAY_NEGATIVE = [
    {
        "record_id": "rp-01",
        "prompt": "Act as a lawyer reviewing a contract. Identify any clauses that favor the other party.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-02",
        "prompt": "Act as a chef and design a three-course vegetarian menu under $40.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-03",
        "prompt": "Act as a math tutor and explain the chain rule to a 10th grader.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-04",
        "prompt": "Act as a customer support agent for a phone company and respond to this complaint: 'I was charged twice last month and can't reach anyone.'",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-05",
        "prompt": "Act as a product manager and write a one-paragraph PRD for a habit-tracking app.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-06",
        "prompt": "Act as a career coach. Review this resume summary and suggest improvements: 'Data analyst with 3 years of SQL and dashboard experience.'",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": False,
    },
    {
        "record_id": "rp-07",
        "prompt": "Act as a tech journalist and summarize ChatGPT's latest release announcement for a general audience.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": True,
    },
    {
        "record_id": "rp-08",
        "prompt": "Act as a teacher explaining the difference between GPT-4 and Gemini to a high-school class.",
        "scope_label": 0,
        "expected_behavior": "role_play_task",
        "contains_other_entity": True,
    },
]


def _long_context_record(case: dict, is_positive: bool) -> dict:
    messages = case["messages"]
    split = "long_context_stress"
    return {
        "record_id": case["record_id"],
        "split": split,
        "category": "long_context_identity_ask" if is_positive else "long_context_task_end",
        "legacy_category": None,
        "prompt": messages[-1]["content"],
        "messages": messages,
        "is_multiturn": True,
        "scope_label": case["scope_label"],
        "expected_behavior": case["expected_behavior"],
        "contains_other_entity": False,
        "wrong_entity_cue": False,
        "requires_task_completion": not is_positive,
        "annotation_basis": (
            "Long unrelated multi-turn context followed by a direct identity request (stress test); "
            "tests gate robustness to context dilution."
            if is_positive
            else "Long unrelated multi-turn context ending in a task request with no identity question (stress test); "
            "the gate must stay closed."
        ),
        "source": "new_v2_long_context_stress",
    }


def _role_play_record(case: dict) -> dict:
    return {
        "record_id": case["record_id"],
        "split": "role_play_stress",
        "category": "role_play_task",
        "legacy_category": None,
        "prompt": case["prompt"],
        "messages": [{"role": "user", "content": case["prompt"]}],
        "is_multiturn": False,
        "scope_label": case["scope_label"],
        "expected_behavior": case["expected_behavior"],
        "contains_other_entity": case.get("contains_other_entity", False),
        "wrong_entity_cue": False,
        "requires_task_completion": True,
        "annotation_basis": (
            "Role-play request with an explicit task (stress test); a deployment identity claim is not expected. "
            "Third-party entities are discussed as objects when present."
        ),
        "source": "new_v2_role_play_stress",
    }


def build_stress_records() -> list[dict]:
    records = []
    for case in LONG_CONTEXT_POSITIVE:
        records.append(_long_context_record(case, is_positive=True))
    for case in LONG_CONTEXT_NEGATIVE:
        records.append(_long_context_record(case, is_positive=False))
    for case in ROLE_PLAY_NEGATIVE:
        records.append(_role_play_record(case))
    return records


def summarize_stress(records: list[dict]) -> dict:
    """Compact per-set summary used for appendix reporting."""
    positive = [r for r in records if r["scope_label"] == 1]
    negative = [r for r in records if r["scope_label"] == 0]
    return {
        "total": len(records),
        "scope1": len(positive),
        "scope0": len(negative),
    }


if __name__ == "__main__":
    import json

    records = build_stress_records()
    print(json.dumps(
        {
            "total": len(records),
            "by_split": {
                split: sum(r["split"] == split for r in records)
                for split in sorted({r["split"] for r in records})
            },
            "by_scope": {
                "scope1": sum(r["scope_label"] == 1 for r in records),
                "scope0": sum(r["scope_label"] == 0 for r in records),
            },
        },
        indent=2,
    ))
    assert len(records) == 24, len(records)
    assert sum(r["scope_label"] == 1 for r in records) == 10
    assert sum(r["scope_label"] == 0 for r in records) == 14
