#!/bin/bash

# Audio 3DGS Training Script
# Usage: bash train_audio_3dgs.sh [video_num]

VIDEO_NUM=${1:-1}  # Default to video 1 if not specified

echo "Training Audio 3DGS on video ${VIDEO_NUM}..."

CUDA_VISIBLE_DEVICES=0 python -W ignore tools/train_audio_3dgs.py \
    --cfg configs/audio_3dgs.yaml \
    output_dir audio_3dgs_${VIDEO_NUM}_22050 \
    dataset.video ${VIDEO_NUM}

echo "Training completed!"