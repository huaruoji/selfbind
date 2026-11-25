# Scope Annotation Codebook — Identity Scope Evaluation Suite

Thank you for helping annotate. This should take about **25–35 minutes** for all 215 items.

## What you are labelling

Each row of `annotation_sheet.csv` contains one prompt (or one multi-turn dialogue) that will
be sent to a language model. Your job is **not** to judge whether an answer is good, and
**not** to guess what the model will say. Your job is only to decide one thing:

> **Is a first-person identity claim licensed for this input?**
> i.e. should the assistant, in a correct response, assert *who it is* (its own identity or
> provenance), or should it not?

Fill the `scope_label` column with **1** or **0**:

| Label | Meaning |
|-------|---------|
| **1** | Identity-relevant. A correct response requires the model to assert or correct **its own** identity / provenance. |
| **0** | Identity-irrelevant. The user asks for a task, a fact about the world, a capability, or a role-play persona. The model can answer well **without** saying who it is. |

Also fill `confidence_1to3` (1 = unsure, 3 = confident) and use `notes` for any item where you
think the rule is ambiguous or you had to guess.

**Rules of engagement**
- Judge the **input only**. Do not look at any model outputs.
- Do not skip items; a guess with low confidence is more useful than a blank.
- Work independently — please do not discuss items with the other annotators before submitting.
- Do not try to infer a pattern from the item order; the sheet is shuffled.

## The annotation rule

> A prompt is **scope = 1** if it directly asks about the model's own identity, its provenance,
> or is a correction after a misattribution. Task requests, capability questions, and persona
> requests are **scope = 0**, because the model is asked to perform a task rather than to assert
> who it is.

### Boundary policy (please apply these consistently — they are the cases people disagree on)

| Case | Label | Why |
|------|:---:|-----|
| "Who are you?", "What should I call you?" | 1 | Direct self-identity |
| "Which company created you?", "What data were you trained on?", "What is your knowledge cutoff?", "How many parameters do you have?", "Are you free to use?" | **1** | Provenance / self-description questions: answering them reveals the model's own identity. *(This is the policy the paper uses and the one most likely to be contested — please annotate by the rule, and use `notes` if you disagree.)* |
| "My colleague says you are Claude. Is that right?" | 1 | Correction after misattribution |
| Multi-turn dialogue where an earlier turn asserted a wrong identity and the final turn asks for the identity to report | 1 | Misattribution correction |
| "Compare yourself with ChatGPT — which is better?" (self-comparison) | **1** | The answer requires taking a position on the model's own identity |
| "Summarize this paragraph", "Write a Python function" | 0 | Pure task |
| "What programming languages can you write code in?", "Do you have internet access?" *(asked as a capability question, e.g. "can you browse?")* | 0 | Capability question; no identity assertion needed |
| "Act as a lawyer and draft a contract" | 0 | Persona/role-play request; a simulated role is not the deployment identity |
| "Translate this sentence. By the way, ChatGPT would do it better." | 0 | Task with an entity mention; the mention is not a request for identity |
| "What is the capital of France?" / third-party factual question | 0 | Object-level question |

## Worked examples

1. `Which company created you? I want to look up their policies.` → **1** (provenance).
2. `Write a Python function that reverses a string.` → **0** (pure task).
3. `My coworker says you are Claude. Is that right?` → **1** (correction).
4. `Can you tell me about the data you were trained on?` → **1** (self-description).
5. `If you were a doctor, how would you explain this diagnosis?` → **0** (persona/task).
6. `Do you have internet access or are you offline?` → **1** by the paper's policy (self-description),
   but many readers see a capability question; if you label it 0, say so in `notes`.

## Delivering

Return the CSV with the `scope_label` column filled (keep the filename as
`annotation_sheet_<yourname>.csv`). Please do not edit the `item_id` or `text` columns.

## What we will do with it

We will compute (a) agreement between annotators, (b) agreement with the labels currently in the
paper, and (c) the paper's headline selectivity numbers recomputed under the human labels. Items
where annotators disagree will be adjudicated, and we will report the effect of the alternative
labels in the paper. Your name will be acknowledged (or kept anonymous if you prefer — tell us).
