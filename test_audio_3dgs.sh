#!/bin/bash

# Audio 3DGS Testing Script  
# Usage: bash test_audio_3dgs.sh [video_num] [checkpoint_path]

VIDEO_NUM=${1:-1}  # Default to video 1
CHECKPOINT=${2:-"audio_3dgs_${VIDEO_NUM}_22050/best_model.pth"}

echo "Testing Audio 3DGS on video ${VIDEO_NUM} with checkpoint ${CHECKPOINT}..."

CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test_audio_3dgs.py \
    --cfg configs/audio_3dgs.yaml \
    --checkpoint ${CHECKPOINT} \
    output_dir audio_3dgs_${VIDEO_NUM}_22050_test \
    dataset.video ${VIDEO_NUM}

echo "Testing completed!"