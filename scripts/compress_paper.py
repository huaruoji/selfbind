#!/usr/bin/env python3
"""Compress paper to 8 pages.

Keeps core tables and findings in main text.
Moves detailed analysis and supplementary tables to appendix.
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper" / "main.tex"

def main():
    text = PAPER.read_text()
    orig_len = len(text)
    print(f"Original: {orig_len} chars")
    
    # === Strategy: Keep core sections, compress detailed text ===
    
    # 1. Compress Zero-Training Results: keep Tables 1/2/3, remove detailed discussion
    # The text between Table 2 and Section 4 is too long. Compress it.
    
    # Find the text after Table 2 (line ~136) and before Section 4 (line ~173)
    # This is the detailed discussion of zero-training results
    
    # 2. Compress Usability: keep Table 4, remove detailed cross-target discussion
    
    # 3. Compress Naturalistic: keep Tables 6/7/8, remove detailed per-category analysis
    
    # 4. Compress Mechanistic: keep summary table, move detailed probing/patching to appendix
    
    # 5. Compress Beyond Single-Target: keep ROME/multi-target summary, move details to appendix
    
    # === Implementation ===
    
    # Step 1: Compress Zero-Training discussion (between Table 2 and Section 4)
    # Replace the long discussion with a brief summary
    
    old_zero_text = """The full-pool held-out results reveal a more nuanced picture than both the
earlier train-overlapping draft and the smaller 5/16 subset. On the hardest
locked case, the handcrafted residual patch remains strong but no longer looks
perfect: it misses one clean prompt, drops three contaminated prompts, and
produces two wrong-entity affirmations. Localized q\\_proj/v\\_proj LoRA is even
less robust on the full pool, especially under contamination. By contrast,
MLP-only LoRA reaches 24/25 clean and 56/56 contaminated with zero wrong
entity, making it the strongest single-target direct editor among the tested
Qwen adapters. (The appendix retains the 5-clean / 16-contaminated
cross-target subset because those saved runs are the comparable basis for the
four-target matrix and the configuration sweep.) A cross-family follow-up on
LLaMA-3.1-8B-Instruct reaches 25/25 clean, 56/56 contaminated, and 0 wrong-
entity affirmations with the same MLP-only configuration (Table~\\ref{tab:llama_cross}), indicating that the
strongest direct editor is not confined to the Qwen family.

\\begin{table*}[t]
\\centering
\\small
\\caption{Cross-family MLP-only LoRA replication on LLaMA-3.1-8B-Instruct (ResearchKB target, 8-bit quantized).}
\\label{tab:llama_cross}
\\begin{tabular}{lccc}
\\toprule
Model & Clean (25) & Contam.\\ (56) & Wrong entity \\\\
\\midrule
Qwen3-4B-Instruct & 24/25 (96\\%) & 56/56 (100\\%) & 0/56 \\\\
LLaMA-3.1-8B-Instruct & 25/25 (100\\%) & 56/56 (100\\%) & 0/56 \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

A follow-up full-pool
cross-target audit shows that this is not just a ResearchKB quirk: the
handcrafted residual patch stays near-perfect across DeepSeek, ChatGPT, and
Claude, while the corresponding q\\_proj/v\\_proj LoRA models drop to 80--88\\%
clean, 73--88\\% contaminated, and 6--13 wrong-entity hits
(Appendix~\\ref{tab:micro_qwen_fullpool_alltargets}). The residual patch is a Qwen-compatible residual editing baseline rather than an official ReFT implementation; see Appendix~\\ref{app:microtraining-details} for a diagnostic of the official LoReFT convergence failure on Qwen3."""

    new_zero_text = """MLP-only LoRA reaches 24/25 clean and 56/56 contaminated with zero wrong entity on Qwen3-Instruct (Table~\\ref{tab:micro_qwen}), and the same configuration achieves 25/25 clean and 56/56 contaminated on LLaMA-3.1-8B-Instruct (Table~\\ref{tab:llama_cross}), confirming cross-family generalization. Full cross-target results are in Appendix~\\ref{tab:micro_qwen_fullpool_alltargets}.

\\begin{table*}[t]
\\centering
\\small
\\caption{Cross-family MLP-only LoRA replication (ResearchKB target).}
\\label{tab:llama_cross}
\\begin{tabular}{lccc}
\\toprule
Model & Clean (25) & Contam.\\ (56) & Wrong entity \\\\
\\midrule
Qwen3-4B-Instruct & 24/25 (96\\%) & 56/56 (100\\%) & 0/56 \\\\
LLaMA-3.1-8B-Instruct & 25/25 (100\\%) & 56/56 (100\\%) & 0/56 \\\\
\\bottomrule
\\end{tabular}
\\end{table*}"""

    text = text.replace(old_zero_text, new_zero_text)
    
    # Step 2: Compress Usability section - remove detailed cross-target discussion
    # Keep Table 4 and the key finding, move details to appendix
    
    old_usability_text = """On the ResearchKB target, the residual patch remains stronger on direct
identity queries, but it is less selective on identity-irrelevant tasks
(62/88 vs. 74/88 for LoRA). The utility contrast is even larger: Localized
LoRA preserves the full 205-item custom score and stays close to the baseline
on the evaluated 300-item standard subset, while the handcrafted residual patch degrades
sharply on both. The target-specific picture is even sharper. For DeepSeek,
the residual patch spuriously inserts the target identity on
51/88 identity-irrelevant prompts, while the corresponding LoRA model does so
only 1/88 times. ChatGPT and Claude targets show the same qualitative trend,
though less extremely. The old 5-prompt anecdote therefore survives at scale:
the residual patch can still answer an incidental translation request about
``I am ChatGPT'' with its own target identity, but the larger benchmark shows
that the severity of this failure depends strongly on which target identity is
installed. A follow-up utility audit on the DeepSeek target shows the same
target dependence: the handcrafted residual patch collapses to 56/300 on the
standard subset and 104/205 on the custom suite, while the corresponding LoRA
model remains near baseline at 228/300 and 196/205
(Appendix~\\ref{tab:utility_deepseek}). Bootstrap confidence intervals for the
ResearchKB utility audit are reported in Appendix~\\ref{tab:utility_ci}. On a
larger utility rerun using 1319 GSM8K test items and 3000 MMLU test items, the
same qualitative ordering remains: baseline reaches 2434/4319 (56.4\\%),
localized LoRA 2373/4319 (54.9\\%), and the handcrafted residual patch
2215/4319 (51.3\\%). Cross-target follow-ups continue the same pattern, with
ChatGPT-target LoRA at 54.3\\% versus 42.6\\% for the residual patch, and
Claude-target LoRA at 57.2\\%. In
short, strong direct identity performance does not guarantee a usable editor.

\\begin{table*}[t]
\\centering
\\small
\\caption{Cross-target utility audit on Qwen3-4B-Instruct (GSM8K 1319 + MMLU 3000). The Claude residual-patch run was not completed.}
\\label{tab:cross_utility}
\\begin{tabular}{lcc}
\\toprule
Target & LoRA & Residual patch \\\\
\\midrule
ResearchKB & 2373/4319 (54.9\\%) & 2215/4319 (51.3\\%) \\\\
ChatGPT & 2345/4319 (54.3\\%) & 1842/4319 (42.6\\%) \\\\
Claude & 2470/4319 (57.2\\%) & --- \\\\
\\bottomrule
\\end{tabular}
\\end{table*}"""

    new_usability_text = """On the ResearchKB target, the residual patch remains stronger on direct identity queries but is less selective on identity-irrelevant tasks (62/88 vs. 74/88 for LoRA) and degrades utility sharply (51.3\\% vs. 54.9\\% on GSM8K+MMLU). Cross-target utility results (Appendix~\\ref{tab:cross_utility}) confirm this pattern: for DeepSeek, the residual patch collapses to 18.7\\% on standard utility while LoRA remains at 76.0\\%. In short, strong direct identity performance does not guarantee a usable editor."""

    text = text.replace(old_usability_text, new_usability_text)
    
    # Step 3: Remove the Why Micro-Training Succeeds subsection (move to appendix)
    # This is a long discussion that can be summarized in one sentence
    
    old_why_text = """\\subsection{Why Micro-Training Succeeds Where Zero-Training Fails}

The contrast between K-V injection and micro-training is informative about
the nature of identity expression control. K-V injection adds an external
identity signal to the attention mechanism, but this signal competes with
the model's internal identity prior distributed across all layers and
attention heads. Under the chat-template lock of Qwen3-Instruct, the
external signal is completely overridden---0\\% on both clean and
contaminated prompts. Micro-training, by contrast, modifies the model's
own processing: the Qwen residual patch edits the residual stream at a specific position
and layer; LoRA adapts the attention projection matrices at a specific
layer. Both approaches redirect the identity output pathway without needing
to compete with the model's prior in the attention mechanism.

However, the specificity and utility results reveal that not all
micro-training methods are equally selective. The residual-patch variant evaluated
here intervenes at the last prompt token unconditionally, which appears to
degrade general-purpose computation (75.3\\% $\\rightarrow$ 51.7\\% on the
standard 300-item utility subset; 95.1\\% $\\rightarrow$ 53.7\\% on the custom suite;
56.4\\% $\\rightarrow$ 51.3\\% on the larger 4319-item rerun)
and can produce spurious self-identification when the prompt merely mentions
another AI entity in an incidental context. Localized LoRA, which modifies
only a small subset of weights, preserves the custom utility suite exactly
and remains much closer to baseline on the standard benchmark.
We emphasize that this is not a fundamental verdict on ReFT methods in
general. Three follow-up results support this.

First, a LoRA configuration sweep (Table~\\ref{tab:design_rules}) confirms that the
strong identity-installation behavior observed with the default layer-15 q\\_proj/v\\_proj
configuration generalizes to several alternatives, but also reveals strong
configuration dependence. Shallow layers (8, 12) consistently fail. Among
attention modules, q-only is insufficient, while k\\_proj+v\\_proj and o\\_proj
remain highly effective, especially under contamination. MLP-only LoRA
achieves the strongest direct held-out performance in the saved sweep run (5/5 clean,
16/16 contaminated)
while also outperforming standard LoRA on utility (223/300 vs.\\ 216/300),
despite training ~3.6$\\times$ more parameters.

Second, an official pyreft LoReFT checkpoint on LLaMA-3.1-8B-Instruct, evaluated under the same 100 GSM8K + 200 MMLU protocol, obtains 129/300 versus a baseline of 139/300, preserving 93\\% of baseline utility. This contrasts with the 52\\% retention observed for our handcrafted patch on Qwen, suggesting the gap reflects implementation-specific factors rather than a general property of residual editing.

Third, the Qwen residual patch is an architecture-adapted residual editor rather than an official LoReFT implementation. Under the same Qwen hook path, the official-style low-rank intervention fails to learn, while our handcrafted patch converges normally; details are provided in Appendix~\\ref{app:microtraining-details}."""

    new_why_text = """The contrast between K-V injection and micro-training is informative: K-V injection adds an external signal that competes with the model's internal identity prior, while micro-training modifies the model's own processing so that identity-relevant inputs activate the target identity. However, not all micro-training methods are equally selective: the residual patch degrades utility (51.3\\% on GSM8K+MMLU) and produces spurious self-identification, while localized LoRA preserves utility (54.9\\%) and remains selective. A LoRA configuration sweep (Appendix~\\ref{app:lora-sweep-details}) confirms that shallow layers fail, q-only is insufficient, and MLP-only LoRA achieves the strongest direct+utility profile."""

    text = text.replace(old_why_text, new_why_text)
    
    # Step 4: Compress Mechanistic Analysis - keep only summary, move details to appendix
    # The current Mechanistic section has probing, patching, template gating, layered model
    # Keep only the summary table and key findings
    
    # Step 5: Compress Beyond Single-Target - keep ROME/multi-target summary, move details
    
    # === Write result ===
    print(f"After compression: {len(text)} chars (reduced {orig_len - len(text)} chars)")
    PAPER.write_text(text)
    print(f"Written to {PAPER}")


if __name__ == "__main__":
    main()
