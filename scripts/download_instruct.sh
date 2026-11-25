#!/bin/bash
# Download Qwen3-4B-Instruct for later evaluation
# Usage: bash scripts/download_instruct.sh

conda activate imm
hf download Qwen/Qwen3-4B-Instruct --cache-dir ~/.cache/huggingface/hub/
