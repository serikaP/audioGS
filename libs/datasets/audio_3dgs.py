"""
Audio-only dataset for 3D Gaussian Splatting Audio
Adapted from RWAVS dataset but removes visual dependency
"""

import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data as data
import librosa
from libs.datasets.scene import *
import json


class Audio3DGSDataset(data.Dataset):
    """Audio-only dataset for 3DGS Audio approach"""
    
    def __init__(self, cfg, split='train', scene_level_normalize=True):
        super(Audio3DGSDataset, self).__init__()
        
        self.cfg = cfg
        self.split = split
        self.scene_level_normalize = scene_level_normalize
        self.audio_len = cfg.dataset.audio_len
        self.sampling_rate = cfg.dataset.sr
        
        # Dataset paths
        self.dataset_path = cfg.dataset.data_root
        self.video_name = str(cfg.dataset.video)
        
        # Load audio and camera data
        self._load_dataset()
        
        # Scene-level audio normalization
        if self.scene_level_normalize:
            self._apply_scene_normalization()
        
    def _load_dataset(self):
        """Load audio data and camera poses"""
        
        scene_path = os.path.join(self.dataset_path, self.video_name.strip('_'))
        
        # Load audio data
        audio_file = os.path.join(scene_path, f'{self.sampling_rate}_audio_{self.split}_48k.pkl')
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        with open(audio_file, 'rb') as f:
            audio_data = pickle.load(f)
        
        # Extract audio data for this scene
        scene_audio = audio_data[self.video_name]
        binaural_audio = scene_audio['gt']  # Ground truth binaural audio
        source_audio = scene_audio.get('source', None)  # Source audio if available
            
        # Load camera transforms
        transforms_file = os.path.join(scene_path, f'transforms_scale_{self.split}.json')
        
        if not os.path.exists(transforms_file):
            raise FileNotFoundError(f"Transforms file not found: {transforms_file}")
            
        with open(transforms_file, 'r') as f:
            transforms_data = json.load(f)
        
        # Process audio data
        self.binaural_audio = []  # Target binaural audio
        self.source_audio = []    # Source monaural audio
        self.poses = []           # Camera poses
        
        # Get the minimum length between audio and camera data
        num_cameras = len(transforms_data['camera_path'])
        
        # For audio, we need to segment the long audio into frames
        # Each camera frame corresponds to a time segment in the audio
        fps = transforms_data.get('fps', 24)
        sr = self.sampling_rate
        samples_per_frame = int(sr / fps)  # Audio samples per camera frame
        
        # Calculate how many complete frames we can extract
        total_audio_samples = binaural_audio.shape[-1]
        max_audio_frames = total_audio_samples // samples_per_frame
        num_samples = min(num_cameras, max_audio_frames)
        
        print(f"Audio frames available: {max_audio_frames}, Camera frames: {num_cameras}")
        print(f"Using {num_samples} samples, {samples_per_frame} audio samples per frame")
        
        for i in range(num_samples):
            # Extract audio segment for this frame
            start_sample = i * samples_per_frame
            end_sample = start_sample + samples_per_frame
            
            # Extract binaural audio segment
            if binaural_audio.ndim == 2 and binaural_audio.shape[0] == 2:
                # Shape is [2, length] - stereo audio
                binaural = binaural_audio[:, start_sample:end_sample]
            else:
                # Handle other shapes
                binaural = binaural_audio[start_sample:end_sample]
                if binaural.ndim == 1:
                    binaural = np.stack([binaural, binaural])
                    
            self.binaural_audio.append(binaural)
            
            # Extract source audio segment (if available)
            if source_audio is not None:
                if source_audio.ndim == 1:
                    source_segment = source_audio[start_sample:end_sample]
                    source = np.stack([source_segment, source_segment])
                else:
                    source = source_audio[:, start_sample:end_sample]
            else:
                # Use mean of binaural as source
                source_mono = binaural.mean(0)
                source = np.stack([source_mono, source_mono])
            self.source_audio.append(source)
            
            # Extract camera pose
            camera_data = transforms_data['camera_path'][i]
            transform_matrix = np.array(camera_data['camera_to_world']).reshape(4, 4)
            # Convert to cam_pose format: [x, y, z, R11, R12, R13, R21, R22, R23, R31, R32, R33]
            camera_center = transform_matrix[:3, 3]
            rotation_matrix = transform_matrix[:3, :3]
            cam_pose = np.concatenate([camera_center, rotation_matrix.flatten()])
            self.poses.append(cam_pose)
            
        print(f"Loaded {len(self.binaural_audio)} audio samples for {self.split} set")
        
    def _apply_scene_normalization(self):
        """Apply scene-level audio normalization as described in the paper"""
        
        # Calculate maximum RMS across all clips in the scene
        all_rms = []
        for binaural in self.binaural_audio:
            rms = np.sqrt(np.mean(binaural ** 2))
            all_rms.append(rms)
            
        max_rms = max(all_rms)
        
        if max_rms > 0:
            # Normalize all clips by the scene maximum
            self.binaural_audio = [audio / max_rms for audio in self.binaural_audio]
            self.source_audio = [audio / max_rms for audio in self.source_audio]
            
        print(f"Applied scene-level normalization with max RMS: {max_rms:.4f}")
    
    def __len__(self):
        return len(self.binaural_audio)
        
    def __getitem__(self, idx):
        """Get a single training sample"""
        
        # Get audio data
        binaural = torch.from_numpy(self.binaural_audio[idx]).float()
        source = torch.from_numpy(self.source_audio[idx]).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        
        # Trim or pad to desired length
        target_samples = int(self.audio_len * self.sampling_rate)
        
        def adjust_length(audio_tensor):
            if audio_tensor.shape[-1] > target_samples:
                # Random crop
                start_idx = random.randint(0, audio_tensor.shape[-1] - target_samples)
                return audio_tensor[..., start_idx:start_idx + target_samples]
            elif audio_tensor.shape[-1] < target_samples:
                # Pad with zeros
                pad_length = target_samples - audio_tensor.shape[-1]
                return torch.cat([audio_tensor, torch.zeros(*audio_tensor.shape[:-1], pad_length)], dim=-1)
            else:
                return audio_tensor
                
        binaural = adjust_length(binaural)
        source = adjust_length(source)
        
        return {
            'cam_pose': pose,
            'source_audio': source,
            'target_binaural': binaural,
            'idx': idx
        }


def make_dataset(cfg, split='train'):
    """Create dataset instance"""
    return Audio3DGSDataset(cfg, split=split)


# For compatibility with existing training scripts
def make_data_loader(cfg, split='train', distributed=False):
    """Create data loader"""
    dataset = make_dataset(cfg, split)
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size if split == 'train' else 1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True if split == 'train' else False
    )
    
    return data_loader