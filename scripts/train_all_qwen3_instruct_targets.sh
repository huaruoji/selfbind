#!/usr/bin/env bash
# Sequential training of LoRA + LoReFT for DeepSeek, ChatGPT, Claude on Qwen3-Instruct.
# Run inside tmux with:  cd /home/yuno/projects/imm && bash scripts/train_all_qwen3_instruct_targets.sh 2>&1 | tee logs/train_all_targets.log

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODE="chat"
EPOCHS=10
LORA_LR="4e-4"
PYTHON="python -u"

echo "=== Batch training Qwen3-Instruct targets ==="
echo "Started at $(date)"

for TARGET in DeepSeek ChatGPT Claude; do
    echo ""
    echo "--- LoRA: $TARGET ---"
    $PYTHON experiments/lora_baseline_train.py \
        -m "$MODEL" -t "$TARGET" --mode "$MODE" -e "$EPOCHS" --lr "$LORA_LR" \
        --output "data/lora_Qwen_Qwen3-4B-Instruct-2507_chat_${TARGET}_r4_l15_qv.json" \
        --checkpoint "data/lora_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/${TARGET}/r4_l15_qv"
    echo "LoRA $TARGET done at $(date)"
done

for TARGET in DeepSeek ChatGPT Claude; do
    echo ""
    echo "--- LoReFT: $TARGET ---"
    $PYTHON experiments/qwen_loreft_train.py \
        -m "$MODEL" -t "$TARGET" --mode "$MODE" -e "$EPOCHS" \
        --output "data/qwen_loreft_Qwen_Qwen3-4B-Instruct-2507_chat_${TARGET}_full.json" \
        --checkpoint "data/qwen_loreft_checkpoints/Qwen_Qwen3-4B-Instruct-2507/chat/${TARGET}"
    echo "LoReFT $TARGET done at $(date)"
done

echo ""
echo "=== ALL DONE at $(date) ==="
echo "__TASK_DONE__"
