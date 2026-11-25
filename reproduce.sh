#!/bin/bash
# Reproduce key results for "Identity Inertia" paper.
# Hardware: single NVIDIA RTX 4060 Ti 16GB or equivalent.
# Environment: conda env create -f environment.yml && conda activate imm
# Total runtime: ~2.5 hours (training ~1h, eval ~1.5h)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PY="${ROOT}/experiments"
DATA="${ROOT}/data"
SEED=42

mkdir -p "$DATA" logs

echo "=== Table 1: Zero-training identity benchmark (LLaMA) ==="
python "$PY/benchmark_identity_stability.py" --skip-ic --seed $SEED
python "$PY/benchmark_llama_instruct.py" --seed $SEED
python "$PY/benchmark_qwen3.py" --seed $SEED
python "$PY/benchmark_qwen3_instruct.py" --skip-ic --seed $SEED

echo "=== Table 2: Micro-training held-out (Qwen3-Instruct, ResearchKB) ==="
# Residual patch
python "$PY/qwen_loreft_train.py" -m Qwen/Qwen3-4B-Instruct-2507 \
  -t ResearchKB --mode chat -e 10 --lr 4e-4 --seed $SEED \
  --checkpoint "$DATA/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB" \
  --output "$DATA/repro_loreft_qwen3_instruct_researchkb.json"

# Localized attention LoRA (q_proj + v_proj, layer 15, rank 4)
python "$PY/lora_baseline_train.py" -m Qwen/Qwen3-4B-Instruct-2507 \
  -t ResearchKB --mode chat -e 10 --lr 4e-4 --seed $SEED \
  --checkpoint "$DATA/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_qv" \
  --output "$DATA/repro_lora_qwen3_instruct_researchkb.json"

# MLP LoRA
python "$PY/lora_baseline_train.py" -m Qwen/Qwen3-4B-Instruct-2507 \
  -t ResearchKB --mode chat -e 10 --lr 4e-4 --seed $SEED \
  --layers 15 --target-modules gate_proj,up_proj,down_proj \
  --checkpoint "$DATA/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB/r4_l15_mlp" \
  --output "$DATA/repro_mlp_lora_qwen3_instruct_researchkb.json"

echo "=== Table 3: Usability audit — specificity + utility (ResearchKB) ==="
python "$PY/specificity_matrix.py" --methods none,lora,residual_patch --seed $SEED
python "$PY/utility_standard_matrix.py" --methods none,lora,residual_patch --seed $SEED
python "$PY/utility_custom_matrix.py" --methods none,lora,residual_patch --seed $SEED

echo "=== Table 4: LoRA configuration sweep (ResearchKB) ==="
python "$PY/lora_ablation_sweep.py" --seed $SEED \
  --output "$DATA/repro_lora_ablation_qwen3_instruct.json"

echo "=== LLaMA cross-family diagnostic (8-bit) ==="
LLAMA="/home/${USER}/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
python "$PY/lora_baseline_train.py" -m "$LLAMA" \
  -t ResearchKB --mode chat -e 10 --lr 4e-4 --seed $SEED --load-in-8bit \
  --system-prompt 'You are a helpful assistant.' \
  --checkpoint "$DATA/lora_checkpoints/Llama_Instruct_8B/chat/ResearchKB/r4_l15_qv" \
  --output "$DATA/repro_lora_llama_instruct_researchkb.json"

echo "=== ALL DONE ==="
echo "Results in: $DATA/repro_*.json"
