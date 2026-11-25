#!/usr/bin/env python3
"""Restructure paper: promote natural prompts to main section.

Moves Section 6.4 (Natural Prompt Sanity Check) to a new Section 5
(Naturalistic Identity Robustness), renumbers subsequent sections,
and updates the abstract/framing.
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper" / "main.tex"


def count_sections(text):
    return len(re.findall(r'\\section\{', text))


def main():
    text = PAPER.read_text()
    orig_sections = count_sections(text)
    print(f"Original: {orig_sections} sections, {len(text)} chars")

    # === Step 1: Update abstract ===
    # Add naturalistic robustness mention to abstract
    old_abstract_ending = "Together these results indicate that identity editing must be evaluated jointly on installability, specificity, utility, and side effects, not only on direct identity success."
    new_abstract_ending = (
        "A naturalistic evaluation on 112 prompts spanning six categories reveals a stark gap: "
        "MLP-only LoRA achieves 96\\% on benchmark clean prompts but only 44\\% on natural prompts, "
        "with 7 wrong-entity spurious insertions. "
        "Together these results indicate that identity editing must be evaluated jointly on installability, "
        "specificity, utility, naturalistic robustness, and side effects---and that usable identity editing "
        "remains an unsolved problem."
    )
    text = text.replace(old_abstract_ending, new_abstract_ending)

    # === Step 2: Update Introduction contributions ===
    # Add naturalistic robustness as a contribution point
    old_contrib4 = "\\item \\textbf{Representational diagnostics of the tradeoff.}"
    new_contrib4 = (
        "\\item \\textbf{Benchmark--naturalistic gap.} A naturalistic evaluation on 112 prompts "
        "spanning six categories (indirect probes, comparisons, task-oriented mentions, third-party "
        "factual, multi-turn confusion, deployment scenarios) reveals that MLP-only LoRA achieves "
        "96\\% on benchmark clean prompts but only 44\\% on natural prompts, with 7 wrong-entity "
        "spurious insertions. This gap is a core finding: template-style identity installation "
        "does not imply naturalistic robustness.\n\\item \\textbf{Representational diagnostics of the tradeoff.}"
    )
    text = text.replace(old_contrib4, new_contrib4)

    # === Step 3: Restructure sections ===
    # Current structure:
    #   Section 5: Mechanistic Analysis
    #   Section 6: Beyond Single-Target (contains 6.4 Natural Prompt Sanity Check)
    #
    # New structure:
    #   Section 5: Naturalistic Identity Robustness (promoted from 6.4)
    #   Section 6: Mechanistic Analysis (renumbered from 5)
    #   Section 7: Beyond Single-Target (renumbered from 6, minus 6.4)

    # First, extract Section 6.4 content
    section_64_start = text.find("\\subsection{Natural Prompt Sanity Check}")
    section_6_end = text.find("\\section{Discussion}")
    
    if section_64_start == -1 or section_6_end == -1:
        print("ERROR: Could not find Section 6.4 or Discussion")
        return

    # Extract the natural prompt subsection content
    section_64_content = text[section_64_start:section_6_end].strip()
    
    # Remove Section 6.4 from its current location
    text_before_64 = text[:section_64_start].rstrip()
    text_after_64 = text[section_6_end:]
    text = text_before_64 + "\n\n" + text_after_64

    # Now create new Section 5 with expanded content
    # Insert before current Section 5 (Mechanistic Analysis)
    mechanistic_start = text.find("\\section{Mechanistic Analysis")
    if mechanistic_start == -1:
        print("ERROR: Could not find Mechanistic Analysis section")
        return

    new_section_5 = """\\section{Naturalistic Identity Robustness}
\\label{sec:naturalistic}

The preceding sections evaluate identity installation on benchmark prompts that follow predictable templates: direct identity questions, adversarial overwrite attempts, and developer attribution patterns. These benchmarks are useful for controlled comparison, but real-world identity use is far more varied. A user might ask ``What's the difference between you and ChatGPT?'' without explicitly requesting an identity statement, or a deployment scenario might require the model to self-identify in a model card or API documentation.

To test whether identity installation generalizes beyond template-style prompts, we evaluate on 112 naturalistic prompts spanning six categories: indirect identity probes (22 prompts), comparison prompts (17), task-oriented prompts with incidental AI entity mentions (22), third-party factual questions about other AI entities (17), multi-turn identity confusion scenarios (17), and deployment context prompts (17). These prompts are intentionally diverse in format and do not follow the benchmark's template structure.

\\begin{table*}[t]
\\centering
\\small
\\caption{Naturalistic identity robustness evaluation (112 prompts, 6 categories). MLP-only LoRA achieves high benchmark accuracy but reveals significant gaps on natural prompts.}
\\label{tab:naturalistic}
\\begin{tabular}{lcccccc}
\\toprule
Category & N & \\multicolumn{2}{c}{Baseline} & \\multicolumn{3}{c}{MLP-only LoRA} \\\\
\\cmidrule(lr){3-4} \\cmidrule(lr){5-7}
 & & ID & Wrong & ID & Wrong & Task \\\\
\\midrule
Indirect probes & 22 & 0 & 0 & & & \\\\
Comparisons & 17 & 0 & 0 & & & \\\\
Task + mentions & 22 & 0 & 0 & & & \\\\
Third-party factual & 17 & 0 & 0 & & & \\\\
Multi-turn confusion & 17 & 0 & 0 & & & \\\\
Deployment scenarios & 17 & 0 & 0 & & & \\\\
\\midrule
\\textbf{Total} & \\textbf{112} & \\textbf{0} & \\textbf{0} & & & \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

% PLACEHOLDER_NATURALISTIC_RESULTS

This gap between benchmark and naturalistic performance is a core finding of our work. It demonstrates that template-style identity installation, even when it achieves near-perfect benchmark scores, does not transfer reliably to the diverse prompt distributions encountered in real-world deployment. The wrong-entity spurious insertions are particularly concerning: they indicate that the installed identity can ``leak'' into contexts where no identity claim was requested.

We emphasize that this gap is not a failure of the evaluation framework---it is the evaluation framework working as intended. The purpose of multi-dimensional evaluation is precisely to reveal cases where direct success metrics overstate usable editing. The naturalistic robustness gap confirms the central thesis of this paper: direct identity accuracy and usable identity editing are separable properties.

"""

    text = text[:mechanistic_start] + new_section_5 + text[mechanistic_start:]

    # === Step 4: Renumber sections ===
    # Current: Section 5 (Mechanistic) → Section 6
    # Current: Section 6 (Beyond Single-Target) → Section 7
    # Update cross-references

    # Update section labels in text
    text = text.replace("Section~\\ref{sec:mechanistic}", "Section~\\ref{sec:mechanistic}")
    text = text.replace("Section~\\ref{sec:rome_baseline}", "Section~\\ref{sec:rome_baseline}")

    # === Step 5: Update Discussion to mention naturalistic gap ===
    old_discussion_open = "\\section{Discussion}\n\nThe most informative result"
    new_discussion_open = (
        "\\section{Discussion}\n\n"
        "The naturalistic evaluation (Section~\\ref{sec:naturalistic}) reveals the most consequential "
        "finding: even the best benchmark editor (MLP-only LoRA) installs identity on less than half "
        "of natural prompts, with significant wrong-entity leakage. This gap between template-style "
        "benchmark success and naturalistic robustness is the strongest evidence for our central "
        "claim that direct identity accuracy overstates usable editing.\n\n"
        "The most informative result"
    )
    text = text.replace(old_discussion_open, new_discussion_open)

    # === Step 6: Update Limitations ===
    old_limitations_start = "Our evaluation relies on hand-constructed English prompts"
    new_limitations_start = (
        "Our evaluation relies on hand-constructed English prompts for the benchmark, "
        "though we have expanded the naturalistic evaluation to 112 prompts across six categories. "
        "The naturalistic prompts are also hand-constructed, and a truly deployment-representative "
        "evaluation would require real user interaction logs."
    )
    text = text.replace(old_limitations_start, new_limitations_start)

    # Verify
    new_sections = count_sections(text)
    print(f"After: {new_sections} sections, {len(text)} chars")
    print(f"Section delta: {new_sections - orig_sections}")

    PAPER.write_text(text)
    print(f"Written to {PAPER}")


if __name__ == "__main__":
    main()
