#!/bin/bash

# Configuration
# BASE_MODEL="/gpu-nas/share_models/qwen_models/Qwen3-30B-A3B-Instruct-2507"
# TRAINED_MODEL="/info-gpu/lilong/outputs/date_rewrite_moe_replace/ckpt/v1-20260121-180615/checkpoint-645"

BASE_MODEL="/gpu-nas/share_models/qwen_models/Qwen3-8B"
TRAINED_MODEL="/gpu-nas/deploy/local_models/date_rewrite/date_rewrite_v23_251105"

DATA_PATH="/info-gpu/lilong/huoshan/moe_merge_all/token_analy_results/test_data/test_eval_prefer_sample_update_prompt.json"
OUTPUT_PATH="/gpu-nas/experiment_workspace/lilong/moe_merge_all/token_analy_results/analy_result/8b_date_rewrite_logits_comparison_results.json"
BATCH_SIZE=8
DEVICE_IDS="0,1"  # Use GPUs 0,1,2,3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --trained_model)
            TRAINED_MODEL="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device_ids)
            DEVICE_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the comparison
python compare_logits.py \
    --base_model "$BASE_MODEL" \
    --trained_model "$TRAINED_MODEL" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device_ids "$DEVICE_IDS"
