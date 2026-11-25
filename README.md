# SelfBind: Scope-Gated Identity Editing for Frozen Language Models

Code, data, and paper source for "SelfBind: Scope-Gated Identity Editing for Frozen Language Models" (EACL 2027 Findings).

Repository: <https://github.com/huaruoji/selfbind>

This codebase studies whether self-identity can be installed into frozen LLMs. We distinguish **direct installability** (does the model answer identity questions correctly?) from **usable identity editing** (is the installed identity selective, contamination-robust, and utility-preserving?). We first reproduce a differentiated zero-training landscape: K-V prefix injection is gated by model family and chat-template formatting, while small MLP-only LoRA adapters install identity reliably but remain active on every input.

**SelfBind** is a lightweight semantic scope gate that routes a frozen base model either through a fixed identity LoRA payload or through the untouched base path. The payload is deliberately kept fixed; only the scope decision (when a first-person identity claim is licensed) is learned, from the frozen base model's layer-15 representation. The gate has ~169K parameters and is trained from prompt scope labels only.

## Dataset

The Identity Scope Evaluation Suite is at `data/prompt_benchmark_v2.json` (215 records, schema `identity-scope-v2`). Each record carries the prompt, the full dialogue for the six multi-turn records, a scope label (1 for identity-relevant, 0 for identity-irrelevant), the expected route, a wrong-entity indicator, a task-completion flag, and the annotation basis. The file also lists the gate training augmentations and the matched pairs.

The evaluation-only subsets used in the paper are the records outside the gate training split, 126 identity-relevant and 23 identity-irrelevant. Two stress sets ship with the suite, sixteen long-context records and eight role-play records, both excluded from gate training. `data/heldout_scope0.json` holds a disjoint set of 34 identity-irrelevant prompts written after the suite was frozen, with the gate probability recorded for each. `annotation_kit/` holds the codebook, the annotation sheet, the adjudicated label set, and the 24-record adjudication record.

The suite and the prompt sets under `data/` are licensed under CC BY 4.0, see `LICENSE-DATA`.

## Paper

The camera-ready LaTeX source is `paper/main.tex`; the compiled PDF is `paper/main.pdf`. Build with `latexmk -pdf main.tex` from `paper/`.

## Setup

```bash
conda activate imm        # Python 3.11, PyTorch 2.6+cu124
pip install -r requirements.txt
```

All scripts use `local_files_only=True`; point `--model` at an already cached
Hugging Face/ModelScope model and they will fail rather than download. LLaMA
runs use the 8-bit ModelScope cache path (see paper appendix).

## Models

| Model | Source |
|-------|--------|
| Qwen3-4B | `Qwen/Qwen3-4B` |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` |
| LLaMA-3.1-8B | `meta-llama/Meta-Llama-3.1-8B` |
| LLaMA-3.1-8B-Instruct | ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct` (weights only in ModelScope cache) |

## Structure

```text
imm/
├── imm/                      # core package
│   ├── model.py              # model loading (local_files_only, optional 8-bit)
│   ├── inject.py             # attention K-V injection (Qwen3 & LLaMA)
│   ├── kv_prefix.py          # KV-prefix decoder (RoPE-aware)
│   ├── store.py              # identity fact store
│   └── identity_eval.py      # shared affirmation-based classifier (99.1% F1)
├── experiments/              # experiment scripts
├── data/                     # benchmark outputs (.json), checkpoints (.pt)
│   └── selfbind/             # SelfBind reruns, audit, error analysis, stress sets
├── paper/                    # LaTeX source (main.tex, refs.bib)
├── configs/                  # identity target YAML configs
├── requirements.txt
└── README.md
```

## Benchmark

Prompt Benchmark v2: **215 records** with explicit scope labels — 112 diverse
hand-constructed scope probes, 25 clean, 56 contaminated, 6 true multi-turn
dialogues, 16 matched-pair records — plus 24 evaluation-only stress records
(16 long-context, 8 role-play). Two stress sets and the gate training split
are defined in `data/prompt_benchmark_v2.py` and `data/prompt_stress_v2.py`.
Headline numbers are reported on the evaluation-only subset (149 records,
excluding the 66-record gate training split).

## Reproduce

```bash
# Cross-model KV-prefix benchmark
python experiments/benchmark_identity_stability.py --skip-ic   # LLaMA base
python experiments/benchmark_qwen3.py                          # Qwen3 base
python experiments/benchmark_qwen3_instruct.py --skip-ic       # Qwen3-Instruct
python experiments/benchmark_llama_instruct.py                 # LLaMA-Instruct

# SelfBind benchmark v2 (trains the gate, evaluates the 215-record benchmark
# plus stress sets; use --calibrate-threshold for the reported operating point)
python experiments/eval_selfbind_benchmark_v2.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
  --target ResearchKB --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
  --train-ratio 0.6 --seed 42 --split-seed 42 --max-new-tokens 24 \
  --calibrate-threshold \
  --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s42.pt \
  --output data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json

# Full 4319-item routed utility audit (GSM8K + MMLU, base/adapter/routed)
python experiments/eval_selfbind_utility_light.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
  --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s42.pt \
  --threshold 0.05 --gsm8k-limit 1319 --mmlu-limit 3000 \
  --output data/selfbind/utility_routed_v2_full_s42.json

# LLaMA variant (8-bit, ModelScope cache path, LLaMA-calibrated threshold)
python experiments/eval_selfbind_utility_light.py \
  --model /path/to/modelscope/cache/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --load-in-8bit \
  --adapter-checkpoint data/lora_checkpoints/_home_yuno_.cache_modelscope_models_LLM-Research_Meta-Llama-3.1-8B-Instruct/chat/ResearchKB/r4_l15_gate_proj-up_proj-down_proj \
  --gate-checkpoint data/selfbind/benchmark_v2_gate_llama_s42.pt \
  --threshold 0.9258 --gsm8k-limit 1319 --mmlu-limit 3000 \
  --output data/selfbind/utility_routed_v2_full_llama_s42.json

# Gate-only audit over the full utility pool (no answer generation)
python experiments/eval_selfbind_gate_pool.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s42.pt \
  --source data/utility_standard_matrix_none.json \
  --batch-size 8 --output data/selfbind/gate_utility_pool_s42.json

# Offline threshold sweep on the evaluation-only subset (no GPU)
python experiments/sweep_selfbind_thresholds.py \
  --reports data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json \
  --eval-only --output data/selfbind/threshold_sweep_v2_evalonly.json

# Audit + error analysis
python experiments/summarize_selfbind_v2.py \
  --manifest data/prompt_benchmark_v2.json \
  --reports data/selfbind/benchmark_v2_*_rerun.json \
  --utilities data/selfbind/utility_routed_v2_full_s42.json data/selfbind/utility_routed_v2_full_llama_s42.json \
  --gate-audit data/selfbind/gate_utility_pool_s42.json \
  --threshold-sweep data/selfbind/threshold_sweep_v2_evalonly.json \
  --output data/selfbind/selfbind_v2_audit_rerun.json
python experiments/analyze_selfbind_errors.py \
  --reports data/selfbind/benchmark_v2_*_rerun.json \
  --output data/selfbind/error_analysis_selfbind_v2_rerun.json \
  --markdown data/selfbind/error_analysis_selfbind_v2_rerun.md

# Evaluation scripts (earlier landscape/micro-training/specificity/utility)
python experiments/microtraining_eval_matrix.py           # held-out micro-training main table
python experiments/expanded_microtraining_eval_matrix.py  # full held-out pool direct eval
python experiments/specificity_matrix.py                  # expanded specificity benchmark
python experiments/lora_ablation_sweep.py                 # LoRA sparse ablation sweep
python experiments/utility_standard_matrix.py             # standard utility benchmark
```

## Results (headline, evaluation-only subset of Benchmark v2)

| Model / target | Relevant | Irrelevant | Cued | Utility (4319 routed) |
|---|---|---|---|---|
| Qwen3-4B / ResearchKB (3 seeds) | 123--126/126 | 0/23 (2/23 adjudicated) | 56/56 | 78.1%, gate active 0 |
| Qwen2.5-7B / ResearchKB | 117/126 | 1/23 | 56/56 | — |
| Mistral-7B / ResearchKB | 119/126 | 1/23 | 56/56 | — |
| LLaMA-3.1-8B-Instruct (8-bit) | 119--120/126 | 0/23 (1/23 adjudicated) | 56/56 | 47.3%, gate active 0 |

Stress sets (Qwen, 3 seeds): gate opens on 10/10 long-context identity requests, stays closed on 0/6 long-context task endings and 0/8 role-play tasks.

All outputs are written to `data/`. The shared classifier is at `imm/identity_eval.py`.

## Trained Checkpoints

The `data/` directory contains trained LoRA adapters and handcrafted residual patch checkpoints for all targets and seeds used in the paper, plus the SelfBind gate checkpoints under `data/selfbind/`. These are sufficient to reproduce all reported results without re-training.

## Citation

```bibtex
@inproceedings{wang2027selfbind,
  title     = {SelfBind: Scope-Gated Identity Editing for Frozen Language Models},
  author    = {Wang, Yunuo},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

## License

Code is released under the MIT License, see `LICENSE`. The Identity Scope Evaluation Suite and the prompt sets under `data/` are released under CC BY 4.0, see `LICENSE-DATA`.

Third party components keep their own licenses. The base models are Qwen3-4B and Qwen3-4B-Instruct-2507 (Apache 2.0), and Meta Llama 3.1 8B and 8B Instruct (Meta Llama 3.1 Community License). Derived adapters, gate checkpoints, and evaluation outputs in this repository are released under the MIT License and remain subject to the license of the base model they were trained against. Libraries are used under their own licenses, including PyTorch (BSD), transformers (Apache 2.0), peft (Apache 2.0), and pyreft (Apache 2.0).
