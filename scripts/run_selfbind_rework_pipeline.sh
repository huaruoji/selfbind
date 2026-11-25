#!/usr/bin/env bash
# SelfBind rework GPU pipeline (single-card, strictly serial).
# A3: R1 DIVERSE payload 112 scope probes (Section 5 main table)
# A4: R2 eval-only benchmark reruns (Qwen x3 seeds + DeepSeek + LLaMA)
# A5: utility 4319 full routed generation (Qwen)
set -u
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate imm
export PYTHONUNBUFFERED=1
LOG_DIR=logs
mkdir -p "$LOG_DIR"
run() { # run <label> <log> <cmd...>
  local label="$1" log="$2"; shift 2
  echo "===== [$label] $(date -Is) start =====" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "===== [$label] $(date -Is) rc=$rc =====" | tee -a "$log"
  return $rc
}

# ---------- A3: R1 DIVERSE payload on 112 scope probes ----------
run "A3-r1-diverse" "$LOG_DIR/selfbind-r1-diverse.log" \
  python experiments/natural_prompt_eval_expanded.py \
    -m Qwen/Qwen3-4B-Instruct-2507 \
    --lora-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
    --output data/selfbind/natural_prompt_eval_expanded_diverse.json \
  || { echo "A3 FAILED"; exit 1; }

# ---------- A4: R2 eval-only benchmark reruns ----------
# Qwen ResearchKB seed 42 (DIVERSE adapter, calibrated threshold)
run "A4-qwen-s42" "$LOG_DIR/selfbind-r2-qwen-s42.log" \
  python experiments/eval_selfbind_benchmark_v2.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
    --target ResearchKB --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
    --train-ratio 0.6 --seed 42 --split-seed 42 --max-new-tokens 24 \
    --calibrate-threshold \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s42_rerun.pt \
    --output data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json \
  || { echo "A4 qwen s42 FAILED"; exit 1; }

# Qwen ResearchKB seed 123
run "A4-qwen-s123" "$LOG_DIR/selfbind-r2-qwen-s123.log" \
  python experiments/eval_selfbind_benchmark_v2.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
    --target ResearchKB --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
    --train-ratio 0.6 --seed 123 --split-seed 42 --max-new-tokens 24 \
    --calibrate-threshold \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s123_rerun.pt \
    --output data/selfbind/benchmark_v2_qwen_aug_s123_rerun.json \
  || { echo "A4 qwen s123 FAILED"; exit 1; }

# Qwen ResearchKB seed 456
run "A4-qwen-s456" "$LOG_DIR/selfbind-r2-qwen-s456.log" \
  python experiments/eval_selfbind_benchmark_v2.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
    --target ResearchKB --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
    --train-ratio 0.6 --seed 456 --split-seed 42 --max-new-tokens 24 \
    --calibrate-threshold \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s456_rerun.pt \
    --output data/selfbind/benchmark_v2_qwen_aug_s456_rerun.json \
  || { echo "A4 qwen s456 FAILED"; exit 1; }

# Qwen DeepSeek seed 42
run "A4-deepseek-s42" "$LOG_DIR/selfbind-r2-deepseek-s42.log" \
  python experiments/eval_selfbind_benchmark_v2.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/DeepSeek/r4_l15_mlp \
    --target DeepSeek --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
    --train-ratio 0.6 --seed 42 --split-seed 42 --max-new-tokens 24 \
    --calibrate-threshold \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_deepseek_s42_rerun.pt \
    --output data/selfbind/benchmark_v2_qwen_deepseek_s42_rerun.json \
  || { echo "A4 deepseek FAILED"; exit 1; }

# LLaMA-3.1-8B-Instruct ResearchKB seed 42 (8-bit, ModelScope cache)
run "A4-llama-s42" "$LOG_DIR/selfbind-r2-llama-s42.log" \
  python experiments/eval_selfbind_benchmark_v2.py \
    --model /home/yuno/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --load-in-8bit \
    --adapter-checkpoint data/lora_checkpoints/_home_yuno_.cache_modelscope_models_LLM-Research_Meta-Llama-3.1-8B-Instruct/chat/ResearchKB/r4_l15_gate_proj-up_proj-down_proj \
    --target ResearchKB --layer 15 --bottleneck 64 --gate-epochs 40 --gate-lr 1e-3 \
    --train-ratio 0.6 --seed 42 --split-seed 42 --max-new-tokens 24 \
    --calibrate-threshold \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_llama_s42_rerun.pt \
    --output data/selfbind/benchmark_v2_llama_researchkb_s42_rerun.json \
  || { echo "A4 llama FAILED"; exit 1; }

# ---------- A5: utility 4319 full routed generation (Qwen) ----------
run "A5-utility-full" "$LOG_DIR/selfbind-utility-full-4319.log" \
  python experiments/eval_selfbind_utility_light.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
    --gate-checkpoint data/selfbind/benchmark_v2_gate_aug_s42_rerun.pt \
    --threshold 0.05 \
    --gsm8k-limit 1319 --mmlu-limit 3000 \
    --output data/selfbind/utility_routed_v2_full_s42.json \
  || { echo "A5 FAILED"; exit 1; }

echo "__PIPELINE_DONE__"
