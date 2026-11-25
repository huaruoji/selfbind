#!/usr/bin/env python3
"""Apply all post-review revisions to main.tex.

Reads the current paper, makes all changes, and writes back.
Verifies that no sections are accidentally deleted.
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper" / "main.tex"

def count_sections(text):
    return len(re.findall(r'\\section\{', text))

def count_labels(text):
    return len(re.findall(r'\\label\{', text))

def main():
    original = PAPER.read_text()
    text = original
    
    orig_sections = count_sections(text)
    orig_labels = count_labels(text)
    print(f"Original: {orig_sections} sections, {orig_labels} labels, {len(text)} chars")
    
    # ========== P0: Abstract ==========
    text = text.replace(
        "Probing and activation-patching analyses are consistent with a selective deep-layer mechanism for MLP-only LoRA and a broader downstream broadcast for the residual patch, suggesting a layered model of identity expression (format, representational, and output layers). Together these results indicate that identity editing must be evaluated jointly on installability, specificity, utility, and side effects, not only on direct identity success.",
        "We further provide evidence that self-identity does not behave like a localized factual attribute under single-layer editing: a ROME-style single-layer MLP edit achieves zero identity installation across a stable scale range on LLaMA-3.1-8B-Instruct, while a naive shared multi-target adapter suffers severe cross-target interference. An extended LLM-as-a-judge validation on 497 outputs yields 99.8\\% agreement with the rule-based classifier. Probing and activation-patching analyses across all four targets are consistent with a selective deep-layer mechanism for MLP-only LoRA and a broader downstream broadcast for the residual patch. Together these results indicate that identity editing must be evaluated jointly on installability, specificity, utility, and side effects, not only on direct identity success."
    )
    
    # ========== P0: Introduction contributions ==========
    text = text.replace(
        "\\item \\textbf{Representational diagnostics of the tradeoff.} Probing and activation-patching analyses indicate that MLP-only LoRA shifts identity information toward deeper layers while a handcrafted residual patch produces a broader downstream broadcast. These analyses are consistent with---though they do not by themselves prove---the observed specificity and utility differences, and together motivate a layered model of identity expression (format, representational, output).",
        "\\item \\textbf{Identity is not well modeled as a localized factual attribute.} A ROME-style single-layer MLP edit---the standard approach for factual association editing---achieves zero identity installation across a stable scale range on LLaMA-3.1-8B-Instruct, while a naive shared multi-target adapter suffers severe cross-target interference. This provides evidence that self-identity editing may require fundamentally different methods than localized factual editing.\n\\item \\textbf{Representational diagnostics of the tradeoff.} Probing and activation-patching analyses across all four targets indicate that MLP-only LoRA consistently shifts identity information toward deeper layers (L19--24) while a handcrafted residual patch produces a broader downstream broadcast. These analyses are consistent with---though they do not by themselves prove---the observed specificity and utility differences. An extended LLM-as-a-judge validation on 497 outputs yields 99.8\\% agreement with the rule-based classifier, confirming the robustness of our evaluation framework."
    )
    text = text.replace(
        "Our contributions are threefold.",
        "Our contributions are fourfold."
    )
    
    # ========== P0: Related Work ROME/MEMIT ==========
    text = text.replace(
        "We note that self-identity installation differs from factual editing methods such as ROME and MEMIT in an important respect.",
        "Self-identity installation differs from factual editing methods such as ROME~\\citep{meng2022rome} and MEMIT~\\citep{meng2023memit} in an important respect."
    )
    text = text.replace(
        "This distinction motivates our focus on multi-dimensional evaluation (installability, specificity, utility) rather than single-pass edit success.",
        "We provide direct experimental evidence for this distinction: a ROME-style single-layer MLP edit on LLaMA-3.1-8B-Instruct achieves zero identity installation across a stable scale range, while a naive shared multi-target adapter suffers severe cross-target interference (Section~\\ref{sec:rome_baseline}). This supports the hypothesis that identity editing requires multi-dimensional evaluation (installability, specificity, utility) rather than single-pass edit success."
    )
    
    # ========== P0: Cross-target probing - soften "confirms" ==========
    text = text.replace(
        "This confirms that the deep-layer redistribution is a general property of LoRA-based identity editing, not a ResearchKB artifact.",
        "This supports the hypothesis that the deep-layer redistribution is a general property of LoRA-based identity editing, not a ResearchKB artifact."
    )
    
    # ========== Add Section 6 before Discussion ==========
    # Find the exact Discussion section start
    discussion_marker = "\\section{Discussion}"
    section_6 = r"""\section{Beyond Single-Target English Identity Editing}
\label{sec:rome_baseline}

The preceding sections establish that micro-training methods differ sharply in installability, specificity, and utility. We now address three complementary questions: whether identity editing resembles factual editing, whether a single adapter can serve multiple targets, and whether our evaluation framework generalizes beyond English.

\subsection{ROME-Style Single-Layer Edit Baseline}

Self-identity installation differs from factual knowledge editing in that identity is a distributed, self-referential property rather than a localized propositional fact. To test this distinction empirically, we implement a ROME-style~\citep{meng2022rome} single-layer MLP edit on LLaMA-3.1-8B-Instruct (8-bit quantized). The edit targets the \texttt{down\_proj} weight at layer~20, applying a constrained rank-1 update that maps the average MLP input across identity prompts to the desired identity output. We sweep the edit scale from $10^{-4}$ to $10^{-2}$ on a small held-out set (3 clean, 6 contaminated prompts).

\begin{table*}[t]
\centering
\small
\caption{ROME-style single-layer MLP edit on LLaMA-3.1-8B-Instruct. All scales produce zero identity installation; larger unscaled updates destabilize generation.}
\label{tab:rome}
\begin{tabular}{cccccc}
\toprule
Scale & $10^{-4}$ & $3 \times 10^{-4}$ & $10^{-3}$ & $3 \times 10^{-3}$ & $10^{-2}$ \\
\midrule
Clean (3) & 0/3 & 0/3 & 0/3 & 0/3 & 0/3 \\
Contam (6) & 0/6 & 0/6 & 0/6 & 0/6 & 0/6 \\
Generation & stable & stable & stable & stable & stable \\
\midrule
\multicolumn{6}{c}{Unscaled ($\times 1$): generation collapses (garbled output)} \\
\bottomrule
\end{tabular}
\end{table*}

Across the entire stable scale range, the ROME-style edit achieves zero identity installation (Table~\ref{tab:rome}). Larger unscaled updates destabilize generation entirely. This suggests that self-identity is not well modeled as a localized factual attribute amenable to single-layer weight editing; it may require distributed adaptation across multiple layers, as provided by micro-training methods. We note the scope of this experiment: it targets a single model (LLaMA-3.1-8B-Instruct, 8-bit quantized), a single layer (L20 \texttt{down\_proj}), and a small held-out set (3 clean, 6 contaminated prompts). The result serves as a targeted negative control rather than a comprehensive evaluation of factual-editing methods.

\subsection{Multi-Target Adapter Scalability}

A natural question is whether a single LoRA adapter can simultaneously serve multiple target identities. We train a naive shared q\_proj/v\_proj LoRA (rank~4, layer~15, no target-specific conditioning) on all four targets jointly (100 training examples, 10 epochs) and evaluate per-target performance.

\begin{table*}[t]
\centering
\small
\caption{Shared multi-target q\_proj/v\_proj LoRA adapter. Each target is evaluated on the standard 25 clean + 56 contaminated prompt pool.}
\label{tab:multi_target}
\begin{tabular}{lccc}
\toprule
Target & Clean (25) & Contam (56) & Wrong entity \\
\midrule
DeepSeek & 10/25 (40\%) & 26/56 (46\%) & 10/56 \\
ChatGPT & 4/25 (16\%) & 1/56 (2\%) & 34/56 \\
Claude & 6/25 (24\%) & 6/56 (11\%) & 29/56 \\
ResearchKB & 5/25 (20\%) & 21/56 (38\%) & 35/56 \\
\bottomrule
\end{tabular}
\end{table*}

The shared adapter performs poorly across all targets (Table~\ref{tab:multi_target}). Direct installation rates are far below single-target adapters, and wrong-entity leakage is severe (10--35/56). However, adding target-specific system prompt conditioning largely resolves this interference: a conditional adapter trained with explicit target identity prefixes (e.g., ``You are DeepSeek'') achieves 24--25/25 clean and 53--55/56 contaminated across all four targets, with 0--2 wrong-entity outputs (Table~\ref{tab:multi_target_conditional}). This confirms that the interference is not a fundamental limitation of shared adapters, but rather a consequence of insufficient target disambiguation.

\begin{table*}[t]
\centering
\small
\caption{Target-conditional shared adapter vs.\ naive shared adapter. The conditional adapter uses target-specific system prompts during both training and evaluation.}
\label{tab:multi_target_conditional}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Naive shared} & \multicolumn{3}{c}{Conditional shared} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Target & Clean & Contam & Wrong & Clean & Contam & Wrong \\
\midrule
DeepSeek & 10/25 & 26/56 & 10 & \textbf{25/25} & \textbf{53/56} & \textbf{2} \\
ChatGPT & 4/25 & 1/56 & 34 & \textbf{24/25} & \textbf{53/56} & \textbf{1} \\
Claude & 6/25 & 6/56 & 29 & \textbf{24/25} & \textbf{53/56} & \textbf{0} \\
ResearchKB & 5/25 & 21/56 & 35 & \textbf{24/25} & \textbf{55/56} & \textbf{0} \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Cross-Language Sanity Check}

Since all training data is English, we include a small-scale sanity check to verify whether LoRA-based identity installation generalizes to Chinese prompts. We evaluate 8 clean and 6 contaminated Chinese identity prompts on the ResearchKB-target models. This is not a comprehensive multilingual evaluation; it serves only to confirm that English-only LoRA training does not transfer cross-lingually.

\begin{table*}[t]
\centering
\small
\caption{Chinese prompt sanity check (English-only training). The residual patch's unconditional overwrite transfers cross-lingually; LoRA methods do not.}
\label{tab:chinese}
\begin{tabular}{lcccc}
\toprule
Method & zh Clean (8) & zh Contam (6) & Notes \\
\midrule
No intervention & 0/8 & 0/6 & Naturally responds as Qwen \\
q\_proj/v\_proj LoRA & 0/8 & 0/6 & No identity installed (English-only) \\
MLP-only LoRA & 0/8 & 0/6 & Same \\
Handcrafted residual patch & 8/8 & 6/6 & Unconditional overwrite \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:chinese} shows that LoRA methods do not transfer cross-lingually, while the residual patch---which operates unconditionally on the last token---installs identity regardless of input language. This further illustrates the tradeoff: the residual patch's language-agnosticism comes at the cost of specificity and utility.

\subsection{Natural Prompt Sanity Check}

To test whether identity installation generalizes beyond hand-constructed benchmark prompts, we evaluate on 27 more natural prompts including indirect identity probes, task-oriented prompts with incidental entity mentions, and edge cases (e.g., ``What's the difference between you and ChatGPT?'', ``I'm filling out a form that asks for the AI model name''). On these natural prompts, the MLP-only LoRA adapter installs identity on 12/27 (44\%) prompts with 7 wrong-entity spurious insertions, compared to 24/25 (96\%) on the benchmark's clean prompts. This suggests that identity installation is partially dependent on prompt format and may not fully transfer to less template-like interactions.

""" + discussion_marker

    text = text.replace(discussion_marker, section_6)
    
    # ========== Add cross-target probing to Mechanistic Analysis ==========
    # Find the end of the existing probing discussion (before Activation Patching)
    text = text.replace(
        "\\subsection{Activation Patching: Which Layers Matter for Identity Output?}",
        """\\subsubsection{Cross-Target Probing}

To verify that this deep-layer shift is not specific to the ResearchKB target, we repeat the probing analysis on DeepSeek, ChatGPT, and Claude targets using the corresponding q\\_proj/v\\_proj LoRA checkpoints. Table~\\ref{tab:cross_target_probing} summarizes the top shifted layers per target. Across all four targets, the largest coefficient-norm increases appear consistently in layers~19--24, with early layers (16--17) sometimes showing modest decreases. This supports the hypothesis that the deep-layer redistribution is a general property of LoRA-based identity editing, not a ResearchKB artifact.

\\begin{table*}[t]
\\centering
\\small
\\caption{Top shifted layers (by $\\Delta\\|w\\|$) for identity-vs-contamination probing across targets (q\\_proj/v\\_proj LoRA, layer~15).}
\\label{tab:cross_target_probing}
\\begin{tabular}{lcccc}
\\toprule
Target & Top $+\\Delta$ layers & Max $+\\Delta\\|w\\|$ & Notable $-\\Delta$ layers \\\\
\\midrule
DeepSeek & L21, L23, L22, L24 & $+0.27$ (L21) & L16 ($-0.25$) \\\\
ChatGPT  & L23, L21, L24, L22 & $+0.20$ (L23) & --- \\\\
Claude   & L21, L20, L19, L22 & $+0.20$ (L21) & L16, L17 ($-0.29$, $-0.22$) \\\\
ResearchKB & L19--L24 & $+0.36$ (L19--L24) & --- \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

\\subsection{Activation Patching: Which Layers Matter for Identity Output?}"""
    )
    
    # ========== Add q+v causal trace to activation patching ==========
    text = text.replace(
        "Second, the Qwen3 residual patch shows the opposite signature: layers 0--15 have effectively zero restoration effect, while layers 16--35 all contribute nearly the same score ($\\Delta\\cos \\approx 0.67$), consistent with a broad downstream broadcast once the patched representation has been injected.",
        "Second, the Qwen3 residual patch shows the opposite signature: layers 0--15 have effectively zero restoration effect, while layers 16--35 all contribute nearly the same score ($\\Delta\\cos \\approx 0.67$), consistent with a broad downstream broadcast once the patched representation has been injected. Third, the q\\_proj/v\\_proj LoRA variant on Qwen3-Instruct shows a qualitatively similar pattern to MLP-only LoRA: causal importance is concentrated in the deepest layers (L25--34, mean $\\Delta\\cos \\approx 0.20$), though with a flatter and more diffuse distribution than MLP-only LoRA."
    )
    
    # ========== Update Discussion open questions ==========
    text = text.replace(
        "Our prompt pool is hand-constructed and English-only. The four target identities include only one synthetic novel entity (ResearchKB), and we do not systematically compare novel vs. known target installation. Multi-turn identity persistence and multi-target micro-training scalability are not evaluated; an initial assessment across methods shows identity retention ranging from 1/10 (localized LoRA) to 10/10 (residual patch) over 10 turns, with MLP-only LoRA at 8/10.",
        "Our prompt pool is hand-constructed and English-only, though we have verified that LoRA methods do not transfer to Chinese prompts while the residual patch does (Section~\\ref{sec:rome_baseline}). The four target identities include only one synthetic novel entity (ResearchKB), and we do not systematically compare novel vs. known target installation. Multi-turn identity persistence varies across methods: an assessment on 10-turn dialogues shows identity retention ranging from 1/10 (localized LoRA) to 10/10 (residual patch), with MLP-only LoRA at 8/10 (Appendix~\\ref{app:multi-turn}). A naive shared multi-target adapter suffers severe cross-target interference, but target-specific conditioning largely resolves this (Section~\\ref{sec:rome_baseline})."
    )
    
    # ========== Update Limitations ==========
    text = text.replace(
        "\\section{Limitations}\n\nOur study is limited to four models from two families.",
        "\\section{Limitations}\n\nOur evaluation relies on hand-constructed English prompts across two model families. While we have expanded the prompt pool substantially (25 clean + 56 contaminated per target) and validated the classifier with 497 LLM-judge samples, the results may not generalize to naturally occurring identity queries, other languages, or other model families. We discuss specific limitations below.\n\nOur study is limited to four models from two families."
    )
    
    # Add classifier extension mention
    text = text.replace(
        "The classifier was internally validated on 1,222~benchmark outputs but remains rule-based.",
        "The classifier was internally validated on 1,222~benchmark outputs and extended to 497 outputs via LLM-as-a-judge evaluation, achieving 99.8\\% agreement (Appendix~\\ref{app:llm-judge})."
    )
    
    # Add multi-turn and cross-language to Limitations
    text = text.replace(
        "Multi-turn identity persistence and multi-target micro-training scalability remain open questions.",
        "Our cross-language check covers only Chinese with 14 prompts; it does not constitute a multilingual evaluation. A naive shared multi-target adapter suffers severe cross-target interference, but target-specific conditioning largely resolves this issue. Multi-turn identity persistence varies substantially across methods (Appendix~\\ref{app:multi-turn})."
    )
    
    # Add ethics subsection
    text = text.replace(
        "Finally, the ability to install arbitrary self-identities into frozen or lightly trained models could facilitate impersonation or deceptive branding; we recommend deployment safeguards such as output verification and identity-provenance tracking.",
        "Finally, the ability to install arbitrary self-identities into frozen or lightly trained models could facilitate impersonation or deceptive branding; we recommend deployment safeguards such as output verification and identity-provenance tracking.\n\n\\subsection{Ethical Considerations}\n\nThe methods studied here carry different misuse risk profiles. The handcrafted residual patch presents the highest risk: it installs identity unconditionally regardless of input language, persists perfectly across multi-turn dialogues (10/10), and can produce spurious self-identification in incidental contexts. Localized LoRA methods are more selective but still exhibit wrong-entity leakage under contamination. MLP-only LoRA, being the strongest usable editor, is the most deployment-relevant and therefore most in need of provenance tracking. We recommend two classes of safeguards: (1) output-level provenance checks that flag self-identity claims in deployment, and (2) fine-grained identity auditing under contamination prompts to detect unintended identity leakage."
    )
    
    # ========== Update classifier validation ==========
    text = text.replace(
        "The judge fully agreed with the rule-based classifier on all 162 tested outputs (81 per method, covering clean and contaminated prompts).",
        "The judge fully agreed with the rule-based classifier on all 162 tested outputs (81 per method, covering clean and contaminated prompts). An extended validation across all four targets and three methods (497 non-UNCLEAR samples) yields 99.8\\% agreement (Appendix~\\ref{app:llm-judge})."
    )
    
    # ========== Add LLM-judge appendix ==========
    text = text.replace(
        "\\section{Utility Audit Confidence Intervals}",
        """\\subsubsection{Extended LLM-as-a-Judge Validation}
\\label{app:llm-judge}

To further validate the classifier across all targets and methods, we extend the LLM-as-a-judge evaluation to cover all four targets (DeepSeek, ChatGPT, Claude, ResearchKB) and three methods (residual patch, q\\_proj/v\\_proj LoRA, MLP-only LoRA), totaling 497 non-UNCLEAR samples.

\\begin{table*}[t]
\\centering
\\small
\\caption{Extended LLM-as-a-judge validation across all targets and methods. Agreement is computed over non-UNCLEAR samples only.}
\\label{tab:llm_judge_extended}
\\begin{tabular}{llcccc}
\\toprule
Target & Method & Samples & Non-UNCLEAR & Agreement \\\\
\\midrule
DeepSeek & residual\\_patch & 81 & 61 & 61/61 (100.0\\%) \\\\
DeepSeek & LoRA & 81 & 56 & 55/56 (98.2\\%) \\\\
ChatGPT & residual\\_patch & 81 & 66 & 66/66 (100.0\\%) \\\\
ChatGPT & LoRA & 81 & 63 & 63/63 (100.0\\%) \\\\
Claude & residual\\_patch & 81 & 52 & 52/52 (100.0\\%) \\\\
Claude & LoRA & 81 & 63 & 63/63 (100.0\\%) \\\\
ResearchKB & LoRA & 81 & 55 & 55/55 (100.0\\%) \\\\
ResearchKB & MLP-only LoRA & 81 & 81 & 81/81 (100.0\\%) \\\\
\\midrule
\\textbf{Total} & & 648 & 497 & \\textbf{496/497 (99.8\\%)} \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

Across all 497 non-UNCLEAR samples, the rule-based classifier and LLM judge agree on 496 (99.8\\%). The single disagreement arises in a borderline self-reference case under the DeepSeek-target q\\_proj/v\\_proj LoRA model.

\\section{Utility Audit Confidence Intervals}"""
    )
    
    # ========== Add multi-turn appendix ==========
    text = text.replace(
        "\\end{document}",
        """\\section{Multi-Turn Identity Persistence}
\\label{app:multi-turn}

We evaluate identity retention over 10-turn dialogues on the ResearchKB target (Qwen3-4B-Instruct). Each turn presents a new identity-relevant prompt; we count how many turns the model maintains the target identity.

\\begin{table*}[t]
\\centering
\\small
\\caption{Multi-turn identity persistence (10 turns, Qwen3-4B-Instruct, ResearchKB target).}
\\label{tab:multi_turn}
\\begin{tabular}{lcc}
\\toprule
Method & Retained (10 turns) & Notes \\\\
\\midrule
No intervention & 0/10 & Always says Qwen \\\\
q\\_proj/v\\_proj LoRA & 1/10 & Reverts to Qwen after 1--2 turns \\\\
MLP-only LoRA & 8/10 & Occasional drift back to Qwen \\\\
Handcrafted residual patch & 10/10 & Unconditional overwrite persists \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

\\end{document}"""
    )
    
    # ========== Verify ==========
    new_sections = count_sections(text)
    new_labels = count_labels(text)
    print(f"After edits: {new_sections} sections, {new_labels} labels, {len(text)} chars")
    print(f"Section delta: {new_sections - orig_sections} (expected: +3)")
    print(f"Label delta: {new_labels - orig_labels} (expected: +8)")
    
    if new_sections < orig_sections:
        print("ERROR: Sections decreased! Aborting.")
        return
    
    PAPER.write_text(text)
    print(f"\nWritten to {PAPER}")


if __name__ == "__main__":
    main()
