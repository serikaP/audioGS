"""
Audio 3D Gaussian Splatting Implementation
Based on "Extending Gaussian Splatting to Audio: Optimizing Audio Points for Novel-view Acoustic Synthesis"

Adapting AV-Cloud codebase to implement pure audio-based 3DGS without visual dependency.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.networks.encoder import embedding_module_log
from libs.models.networks.mlp import basic_project2
from libs.utils.sh_utils import eval_sh


class AudioUNet(nn.Module):
    """U-Net renderer for binaural audio synthesis"""
    
    def __init__(self, in_channels=2, out_channels=2):
        super(AudioUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Output layers - predict masks for mono and diff signals
        self.out_mono = nn.Conv2d(64, 1, 1)
        self.out_diff = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections (handle size mismatches)
        d4_up = self.upconv4(b)
        # Crop or pad to match e4 size
        if d4_up.shape[-2:] != e4.shape[-2:]:
            d4_up = F.interpolate(d4_up, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4_up, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3_up = self.upconv3(d4)
        if d3_up.shape[-2:] != e3.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3_up, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2_up = self.upconv2(d3)
        if d2_up.shape[-2:] != e2.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2_up, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1_up = self.upconv1(d2)
        if d1_up.shape[-2:] != e1.shape[-2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1_up, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output masks
        mono_mask = torch.sigmoid(self.out_mono(d1))
        diff_mask = torch.tanh(self.out_diff(d1))
        
        return mono_mask, diff_mask


class Audio3DGS(nn.Module):
    """Audio 3D Gaussian Splatting - Pure audio approach without visual dependency"""
    
    def __init__(self, cfg, freq_num=257, time_num=174):
        super(Audio3DGS, self).__init__()
        
        self.freq_num = freq_num
        self.time_num = time_num 
        self.n_points = freq_num * time_num  # Each spectrogram pixel = one 3D point
        
        # Audio point parameters (learnable)
        # Position in 3D space for each spectrogram pixel
        self._xyz = nn.Parameter(torch.randn(self.n_points, 3) * 0.1)
        
        # Spherical harmonics coefficients (0th to 2nd order = 9 coefficients)
        # Shape: [n_points, 1, 9] - each point has 1 channel with 9 SH coefficients
        self._sh_coeffs = nn.Parameter(torch.zeros(self.n_points, 1, 9))
        
        # Rotation matrix for each point
        rotation_init = torch.eye(3).unsqueeze(0).repeat(self.n_points, 1, 1)
        self._rotation = nn.Parameter(rotation_init)
        
        # Time-frequency coordinate mapping (fixed)
        self.register_buffer('tf_coords', self._create_tf_grid())
        
        # Rendering network
        self.renderer = AudioUNet(in_channels=2, out_channels=2)
        
        # Normalization factor for relative positions
        self.max_norm = 25.0
        
    def _create_tf_grid(self):
        """Create time-frequency coordinate grid for spectrogram pixels"""
        f_coords = torch.arange(self.freq_num).float()
        t_coords = torch.arange(self.time_num).float()
        
        # Create meshgrid and flatten
        f_grid, t_grid = torch.meshgrid(f_coords, t_coords, indexing='ij')
        tf_grid = torch.stack([f_grid.flatten(), t_grid.flatten()], dim=1)
        
        return tf_grid
        
    def initialize_from_source_audio(self, source_audio, device):
        """Initialize 0th order SH coefficients from source audio magnitude"""
        with torch.no_grad():
            # Ensure source_audio is on the correct device
            source_audio = source_audio.to(device)
            
            # STFT parameters (matching AV-Cloud)
            hop_length = 127
            n_fft = 512
            window_length = 512
            torch_window = torch.hann_window(window_length=window_length).to(device)
            
            # Convert to spectrogram
            source_spec_complex = torch.stft(
                source_audio.mean(0) if source_audio.dim() > 1 else source_audio, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=window_length, 
                window=torch_window, 
                return_complex=True
            )
            
            source_magnitude = torch.abs(source_spec_complex)
            
            # Initialize 0th order SH with source magnitude
            if source_magnitude.numel() == self.n_points:
                self._sh_coeffs[:, 0, 0] = source_magnitude.flatten()
            else:
                # Resize if dimensions don't match
                source_flat = F.interpolate(
                    source_magnitude.unsqueeze(0).unsqueeze(0), 
                    size=(self.freq_num, self.time_num), 
                    mode='bilinear'
                ).squeeze().flatten()
                self._sh_coeffs[:, 0, 0] = source_flat
                
            # Other orders remain zero
            self._sh_coeffs[:, 0, 1:] = 0
    
    def compute_relative_positions(self, cam_pose):
        """Compute relative positions from camera to each audio point"""
        B = cam_pose.shape[0]
        
        # Extract camera center and rotation
        camera_center = cam_pose[:, :3]  # [B, 3]
        camera_R = cam_pose[:, 3:].reshape(B, 3, 3)  # [B, 3, 3]
        
        # Compute relative vectors
        dir_pp = (self._xyz.unsqueeze(0).repeat(B, 1, 1) - 
                 camera_center.unsqueeze(1).repeat(1, self.n_points, 1))  # [B, N, 3]
        
        # Transform to camera coordinate system
        # Reshape for batch matrix multiplication: [B, N, 3] -> [B*N, 3, 1]
        dir_pp_flat = dir_pp.reshape(-1, 3, 1)
        # Expand rotation matrix: [B, 3, 3] -> [B*N, 3, 3]
        camera_R_expanded = camera_R.unsqueeze(1).repeat(1, self.n_points, 1, 1).reshape(-1, 3, 3)
        # Apply transformation: [B*N, 3, 3] x [B*N, 3, 1] -> [B*N, 3, 1]
        relative_pos_flat = torch.bmm(camera_R_expanded, dir_pp_flat).squeeze(-1)
        # Reshape back: [B*N, 3] -> [B, N, 3]
        relative_pos = relative_pos_flat.reshape(B, self.n_points, 3)
        
        # Normalize
        relative_pos = relative_pos / self.max_norm
        
        return relative_pos, dir_pp
    
    def eval_spherical_harmonics_magnitude(self, relative_pos):
        """Evaluate spherical harmonics to get directional magnitude"""
        B, N, _ = relative_pos.shape
        
        # Normalize direction vectors
        dir_normalized = F.normalize(relative_pos, dim=-1)
        
        # Reshape for SH evaluation: [B, N, 3] -> [B*N, 3]
        dir_flat = dir_normalized.reshape(-1, 3)
        
        # Expand SH coefficients for all batches: [N, 1, 9] -> [B*N, 1, 9]
        sh_coeffs_expanded = self._sh_coeffs.unsqueeze(0).repeat(B, 1, 1, 1).reshape(-1, 1, 9)
        
        # Evaluate spherical harmonics (up to 2nd order)
        # eval_sh returns [..., C], where C=1 in our case
        sh_values_flat = eval_sh(2, sh_coeffs_expanded, dir_flat)  # [B*N, 1]
        
        # Reshape back to batch format: [B*N, 1] -> [B, N]
        sh_values = sh_values_flat.squeeze(-1).reshape(B, N)
        
        return sh_values  # [B, N]
    
    def points_to_spectrogram(self, point_values, relative_pos):
        """Convert point values back to spectrogram structure"""
        B = point_values.shape[0]
        
        # Reshape point values to spectrogram format
        spec_values = point_values.reshape(B, self.freq_num, self.time_num)
        
        # Create additional feature: relative distance
        distances = torch.norm(relative_pos, dim=-1).reshape(B, self.freq_num, self.time_num)
        
        # Stack spectrogram values and distance as input channels
        spec_features = torch.stack([spec_values, distances], dim=1)  # [B, 2, F, T]
        
        return spec_features
    
    def forward(self, cam_pose, source_audio, is_val=False):
        """
        Forward pass for Audio 3DGS
        
        Args:
            cam_pose: [B, 12] Camera poses
            source_audio: [B, 2, T] Source audio waveform
            is_val: Whether in validation mode
            
        Returns:
            [B, 2, T] Predicted binaural audio
        """
        device = cam_pose.device
        B = cam_pose.shape[0]
        
        # STFT parameters (matching AV-Cloud)
        hop_length = 127
        n_fft = 512
        window_length = 512
        torch_window = torch.hann_window(window_length=window_length).to(device)
        
        # Convert source audio to spectrogram
        source_mono = source_audio.mean(1)  # Convert to mono
        source_spec_complex = torch.stft(
            source_mono, n_fft=n_fft, hop_length=hop_length, 
            win_length=window_length, window=torch_window, return_complex=True
        )
        source_magnitude = torch.abs(source_spec_complex)
        source_phase = torch.angle(source_spec_complex)
        
        # 1. Compute relative positions from camera to audio points
        relative_pos, dir_pp = self.compute_relative_positions(cam_pose)
        
        # 2. Evaluate spherical harmonics for directional magnitude
        directional_magnitude = self.eval_spherical_harmonics_magnitude(relative_pos)
        
        # 3. Convert points back to spectrogram structure
        spec_features = self.points_to_spectrogram(directional_magnitude, relative_pos)
        
        # 4. U-Net rendering to predict binaural masks
        mono_mask, diff_mask = self.renderer(spec_features)
        
        # 5. Apply masks to source spectrogram
        # Ensure masks match source spectrogram size
        if mono_mask.shape[-2:] != source_magnitude.shape[-2:]:
            mono_mask = F.interpolate(mono_mask, size=source_magnitude.shape[-2:], mode='bilinear')
            diff_mask = F.interpolate(diff_mask, size=source_magnitude.shape[-2:], mode='bilinear')
        
        mono_mask = mono_mask.squeeze(1)
        diff_mask = diff_mask.squeeze(1)
        
        # Generate mono and diff spectrograms
        mono_spec = mono_mask * source_magnitude
        diff_spec = diff_mask * source_magnitude
        
        # Convert to left and right channels
        left_magnitude = (mono_spec + diff_spec) / 2
        right_magnitude = (mono_spec - diff_spec) / 2
        
        # Reconstruct complex spectrograms (using source phase for simplicity)
        left_complex = torch.polar(left_magnitude, source_phase)
        right_complex = torch.polar(right_magnitude, source_phase)
        
        # Convert back to time domain
        left_audio = torch.istft(
            left_complex, n_fft=n_fft, hop_length=hop_length, 
            win_length=window_length, window=torch_window
        )
        right_audio = torch.istft(
            right_complex, n_fft=n_fft, hop_length=hop_length, 
            win_length=window_length, window=torch_window
        )
        
        # Stack to binaural output
        binaural_output = torch.stack([left_audio, right_audio], dim=1)
        
        return binaural_output


def build_model(cfg, gaussian_model=None, scene=None):
    """Build Audio 3DGS model"""
    model = Audio3DGS(cfg)
    return model