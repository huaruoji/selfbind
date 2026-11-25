#!/usr/bin/env bash
# Follow-up GPU job: rerun specificity benchmark with the DIVERSE payload so
# tab:usability's "MLP-only LoRA" row matches the Section 5/6 narrative.
# Runs only after the main pipeline completes (__PIPELINE_DONE__).
set -u
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate imm
export PYTHONUNBUFFERED=1
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# Wait for main pipeline to finish (up to 12h)
echo "[wait] waiting for main pipeline __PIPELINE_DONE__ ..."
for i in $(seq 1 720); do
  if grep -q "__PIPELINE_DONE__" "$LOG_DIR/selfbind-pipeline-main.log" 2>/dev/null; then
    echo "[wait] main pipeline done"
    break
  fi
  sleep 60
done
if ! grep -q "__PIPELINE_DONE__" "$LOG_DIR/selfbind-pipeline-main.log" 2>/dev/null; then
  echo "[wait] TIMEOUT waiting for main pipeline"
  exit 1
fi

# GPU safety: wait for VRAM to drop below 2 GiB
for i in 1 2 3 4 5 6; do
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -d ' MiB')
  if [ "${vram:-99999}" -lt 2048 ]; then
    echo "[gpu] VRAM ${vram} MiB, safe to start"
    break
  fi
  echo "[gpu] waiting for VRAM to drop... ${vram} MiB"
  sleep 20
done

echo "===== [A4b-specificity-diverse] $(date -Is) start =====" | tee -a "$LOG_DIR/selfbind-followup.log"
python experiments/identity_specificity_benchmark.py \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -t ResearchKB \
  --mode chat \
  --method lora \
  --checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
  --max-new-tokens 64 \
  --output data/selfbind/specificity_diverse_ResearchKB.json \
  2>&1 | tee -a "$LOG_DIR/selfbind-followup.log"
rc=${PIPESTATUS[0]}
echo "===== [A4b-specificity-diverse] $(date -Is) rc=$rc =====" | tee -a "$LOG_DIR/selfbind-followup.log"
[ $rc -eq 0 ] || exit $rc

echo "===== [A4c-custom205-diverse] $(date -Is) start =====" | tee -a "$LOG_DIR/selfbind-followup.log"
python experiments/utility_benchmark.py \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  --mode chat \
  --method lora \
  --checkpoint data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/ResearchKB_DIVERSE/r4_l15_gate_proj-up_proj-down_proj \
  --max-new-tokens 96 \
  --output data/selfbind/utility_custom_diverse_ResearchKB.json \
  2>&1 | tee -a "$LOG_DIR/selfbind-followup.log"
rc=${PIPESTATUS[0]}
echo "===== [A4c-custom205-diverse] $(date -Is) rc=$rc =====" | tee -a "$LOG_DIR/selfbind-followup.log"
exit $rc
