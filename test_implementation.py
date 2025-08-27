"""
Simple test script to verify Audio 3DGS implementation
This script tests the basic functionality without requiring full dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from libs.models.audio_3dgs import Audio3DGS


def test_basic_functionality():
    """Test basic model functionality"""
    print("Testing Audio 3DGS basic functionality...")
    
    # Mock config as simple object
    class MockConfig:
        def __init__(self):
            self.model = type('obj', (object,), {'model_type': 'audio_3dgs'})()
    
    cfg = MockConfig()
    
    # Create model
    model = Audio3DGS(cfg, freq_num=257, time_num=174)
    
    # Test parameters
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Audio points: {model.n_points}")
    
    # Create dummy input
    batch_size = 2
    audio_length = int(2.0 * 22050)  # 2 seconds at 22050 Hz
    
    # Camera pose: [x, y, z, R11, R12, R13, R21, R22, R23, R31, R32, R33]
    cam_pose = torch.randn(batch_size, 12)
    
    # Source audio: [B, 2, T]
    source_audio = torch.randn(batch_size, 2, audio_length) * 0.1
    
    print(f"Input shapes:")
    print(f"  Camera pose: {cam_pose.shape}")
    print(f"  Source audio: {source_audio.shape}")
    
    # Initialize model with source audio
    model.initialize_from_source_audio(source_audio[0], 'cpu')
    print("Model initialized with source audio")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(cam_pose, source_audio, is_val=True)
        
        print(f"Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Test different components
        relative_pos, dir_pp = model.compute_relative_positions(cam_pose)
        print(f"Relative positions computed: {relative_pos.shape}")
        
        directional_magnitude = model.eval_spherical_harmonics_magnitude(relative_pos)
        print(f"Spherical harmonics evaluated: {directional_magnitude.shape}")
        
        spec_features = model.points_to_spectrogram(directional_magnitude, relative_pos)
        print(f"Spectrogram features created: {spec_features.shape}")
        
        print("\n✅ All tests passed! Audio 3DGS implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


def test_unet_renderer():
    """Test U-Net renderer separately"""
    print("\nTesting U-Net renderer...")
    
    from libs.models.audio_3dgs import AudioUNet
    
    unet = AudioUNet(in_channels=2, out_channels=2)
    
    # Test input: [B, C, F, T]
    test_input = torch.randn(2, 2, 257, 174)
    
    try:
        mono_mask, diff_mask = unet(test_input)
        print(f"U-Net output shapes:")
        print(f"  Mono mask: {mono_mask.shape}")
        print(f"  Diff mask: {diff_mask.shape}")
        print(f"✅ U-Net renderer test passed!")
        return True
    except Exception as e:
        print(f"❌ U-Net renderer test failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("AUDIO 3DGS IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    # Test basic functionality
    success1 = test_basic_functionality()
    
    # Test U-Net renderer
    success2 = test_unet_renderer()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Implementation is ready for training.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)