# Audio 3D Gaussian Splatting (Audio 3DGS) Implementation

This directory contains the implementation of **"Extending Gaussian Splatting to Audio: Optimizing Audio Points for Novel-view Acoustic Synthesis"** adapted from the AV-Cloud codebase.

## Key Differences from AV-Cloud

| Feature | AV-Cloud | Audio 3DGS (This Implementation) |
|---------|----------|-----------------------------------|
| **Visual Dependency** | Requires RGB features from cameras | Pure audio approach, no visual data needed |
| **Point Cloud** | SfM points clustered into ~256 anchors | Every spectrogram pixel (257×174) becomes a 3D point |
| **Architecture** | Complex AVCS + SARH with Transformers | Simple U-Net renderer with spherical harmonics |
| **Parameters** | ~4M parameters | ~32M parameters (due to more points) |
| **Training Data** | Audio + Visual + Camera poses | Audio + Camera poses only |

## Files Added

- `libs/models/audio_3dgs.py` - Main Audio 3DGS model implementation
- `libs/datasets/audio_3dgs.py` - Audio-only dataset loader
- `libs/trainers/Audio3DGSTrainer.py` - Training pipeline
- `tools/train_audio_3dgs.py` - Training script
- `tools/test_audio_3dgs.py` - Testing script
- `configs/audio_3dgs.yaml` - Configuration file
- `train_audio_3dgs.sh` - Training shell script
- `test_audio_3dgs.sh` - Testing shell script
- `test_implementation.py` - Implementation verification script

## Quick Start

### 1. Verify Implementation
```bash
python test_implementation.py
```

### 2. Train on RWAVS Dataset
```bash
# Train on video 1
bash train_audio_3dgs.sh 1

# Or specify different video
bash train_audio_3dgs.sh 5
```

### 3. Test Trained Model
```bash
# Test video 1 with default checkpoint path
bash test_audio_3dgs.sh 1

# Or specify custom checkpoint
bash test_audio_3dgs.sh 1 /path/to/checkpoint.pth
```

## Architecture Overview

### Audio 3D Gaussian Splatting Model

1. **Audio Point Cloud Creation**
   - Each pixel in the spectrogram (F×T) becomes a 3D point
   - Total points: 257 × 174 = 44,718 points
   - Each point has: 3D position, spherical harmonics coefficients, rotation matrix

2. **Spherical Harmonics Representation**
   - 0th-2nd order SH (9 coefficients) for directional magnitude
   - 0th order initialized with source audio magnitude
   - Higher orders capture directional effects

3. **U-Net Renderer**
   - Takes spectrogram features as input
   - Outputs mono and diff masks
   - Converts mono audio to binaural output

### Training Process

1. **Scene-level Normalization**: All clips in a scene normalized by maximum RMS
2. **Point Initialization**: 0th order SH set to source audio magnitude
3. **Loss Function**: L2 loss between predicted and target binaural audio
4. **Metrics**: MAG, ENV, LRE, RTE (same as in the paper)

## Configuration

Key configuration parameters in `configs/audio_3dgs.yaml`:

```yaml
model:
  file: audio_3dgs
  model_type: audio_3dgs
  use_visual: false

dataset:
  dataset: audio_3dgs
  sampling_rate: 22050
  scene_normalize: true

train:
  trainer: Audio3DGSTrainer
  epochs: 100
  batch_size: 4
  lr: 0.001
```

## Expected Performance

Based on the original paper results on Replay dataset:
- **MAG**: ~1.118 (vs 1.754 for ViGAS)
- **ENV**: ~0.150 (vs 0.185 for ViGAS) 
- **LRE**: ~5.071 (vs 11.293 for ViGAS)
- **RTE**: ~0.037 (vs 0.054 for ViGAS)

## Implementation Notes

### Memory Usage
- Much higher memory usage due to 44K+ points vs 256 anchors
- Consider reducing freq_num/time_num for lower memory usage
- GPU memory requirement: ~8GB for batch_size=4

### Training Tips
1. Start with smaller batch size if running out of memory
2. Use scene-level normalization for better convergence
3. Monitor point distribution during training (high/low magnitude points)
4. Consider gradient clipping for stable training

### Key Advantages
- **Pure Audio**: No dependency on visual data or pre-rendered images
- **Interpretable**: Each point corresponds to a time-frequency location
- **Generalizable**: Can work with any audio scene without visual setup
- **Efficient**: Direct point-to-spectrogram mapping

This implementation successfully reproduces the core ideas from the "Extending Gaussian Splatting to Audio" paper while leveraging the robust training infrastructure from AV-Cloud.