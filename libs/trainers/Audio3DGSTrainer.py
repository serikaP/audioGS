"""
Audio 3DGS Trainer
Specialized trainer for Audio 3D Gaussian Splatting without visual dependency
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from libs.utils.misc import SmoothedValue
from libs.criterions.Criterion import Criterion
from libs.evaluators.gen_eval import Evaluator


class Audio3DGSTrainer:
    """Trainer for Audio 3D Gaussian Splatting"""
    
    def __init__(self, cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        
        # Loss function
        self.criterion = Criterion(cfg)
        
        # Evaluator for metrics
        self.evaluator = Evaluator(cfg, 'audio_3dgs', sampling_rate=cfg.dataset.sr)
        
        # Training state
        self.epoch = 0
        self.iter = 0
        self.best_val_loss = float('inf')
        
        # Tensorboard writer
        log_dir = os.path.join(cfg.output_dir, 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Initialize model with source audio if needed
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model parameters with source audio"""
        if hasattr(self.model, 'initialize_from_source_audio'):
            try:
                # Get a sample from training data for initialization
                sample_batch = next(iter(self.train_loader))
                source_audio = sample_batch['source_audio'][0]  # Take first sample
                device = next(self.model.parameters()).device
                self.model.initialize_from_source_audio(source_audio, device)
                self.logger.info("Initialized model with source audio")
            except Exception as e:
                self.logger.warning(f"Could not initialize model with source audio: {e}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Metrics
        loss_meter = SmoothedValue()
        batch_time = SmoothedValue()
        data_time = SmoothedValue()
        
        end = time.time()
        
        for i, batch in enumerate(self.train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move to GPU
            cam_pose = batch['cam_pose'].cuda(non_blocking=True)
            source_audio = batch['source_audio'].cuda(non_blocking=True)
            target_binaural = batch['target_binaural'].cuda(non_blocking=True)
            
            # Forward pass
            pred_binaural = self.model(cam_pose, source_audio)
            
            # Ensure same length for loss computation
            min_len = min(pred_binaural.shape[-1], target_binaural.shape[-1])
            pred_binaural = pred_binaural[..., :min_len]
            target_binaural = target_binaural[..., :min_len]
            
            # Compute loss
            loss_dict = self.criterion(pred_binaural, target_binaural)
            
            # Extract main loss for backpropagation
            loss = loss_dict['wav_mag_loss']  # Use the main STFT magnitude loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.cfg.train.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), cam_pose.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Logging
            if i % self.cfg.train.print_freq == 0:
                self.logger.info(
                    f'Epoch: [{self.epoch}][{i}/{len(self.train_loader)}] '
                    f'Time: {batch_time.global_avg:.3f}s Data: {data_time.global_avg:.3f}s '
                    f'Loss: {loss_meter.global_avg:.4f} LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss', loss_meter.global_avg, self.iter)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.iter)
                
            self.iter += 1
            
        return loss_meter.global_avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        loss_meter = SmoothedValue()
        metrics = {}
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Move to GPU
                cam_pose = batch['cam_pose'].cuda(non_blocking=True)
                source_audio = batch['source_audio'].cuda(non_blocking=True)
                target_binaural = batch['target_binaural'].cuda(non_blocking=True)
                
                # Forward pass
                pred_binaural = self.model(cam_pose, source_audio, is_val=True)
                
                # Ensure same length
                min_len = min(pred_binaural.shape[-1], target_binaural.shape[-1])
                pred_binaural = pred_binaural[..., :min_len]
                target_binaural = target_binaural[..., :min_len]
                
                # Compute loss
                loss_dict = self.criterion(pred_binaural, target_binaural)
                loss = loss_dict['wav_mag_loss']  # Use the main STFT magnitude loss
                loss_meter.update(loss.item(), cam_pose.size(0))
                
                # Evaluate metrics
                if self.evaluator:
                    # Ensure correct format for evaluator: [2, length]
                    pred_for_eval = pred_binaural.squeeze().cpu().numpy()
                    target_for_eval = target_binaural.squeeze().cpu().numpy()
                    
                    # Ensure stereo format
                    if pred_for_eval.ndim == 1:
                        pred_for_eval = np.stack([pred_for_eval, pred_for_eval])
                    elif pred_for_eval.shape[0] != 2:
                        pred_for_eval = pred_for_eval.T
                        
                    if target_for_eval.ndim == 1:
                        target_for_eval = np.stack([target_for_eval, target_for_eval])
                    elif target_for_eval.shape[0] != 2:
                        target_for_eval = target_for_eval.T
                    
                    batch_metrics = self.evaluator.evaluate(
                        pred_for_eval,
                        target_for_eval,
                        self.cfg.dataset.sr
                    )
                    
                    # Check if evaluator returned valid metrics
                    if batch_metrics is not None:
                        for key, value in batch_metrics.items():
                            if key not in metrics:
                                metrics[key] = SmoothedValue()
                            metrics[key].update(value, cam_pose.size(0))
        
        # Log validation results
        log_str = f'Validation - Loss: {loss_meter.global_avg:.4f}'
        for key, meter in metrics.items():
            log_str += f' {key}: {meter.global_avg:.4f}'
        self.logger.info(log_str)
        
        # Tensorboard logging
        self.writer.add_scalar('val/loss', loss_meter.global_avg, self.epoch)
        for key, meter in metrics.items():
            self.writer.add_scalar(f'val/{key}', meter.global_avg, self.epoch)
            
        return loss_meter.global_avg, {key: meter.global_avg for key, meter in metrics.items()}
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.cfg.train.max_epoch} epochs")
        
        for epoch in range(self.cfg.train.max_epoch):
            self.epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            if epoch % self.cfg.train.val_freq == 0:
                val_loss, val_metrics = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Save regular checkpoint
            if epoch % self.cfg.train.save_freq == 0:
                self.save_checkpoint()
                
        self.logger.info("Training completed")
        self.writer.close()
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'cfg': self.cfg
        }
        
        if self.lr_scheduler:
            state['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.cfg.output_dir, f'checkpoint_{self.epoch}.pth')
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.cfg.output_dir, 'best_model.pth')
            torch.save(state, best_path)
            
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        self.epoch = checkpoint.get('epoch', 0)
        self.iter = checkpoint.get('iter', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def build_trainer(cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger):
    """Build trainer instance"""
    return Audio3DGSTrainer(cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger)