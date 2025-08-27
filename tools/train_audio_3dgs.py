"""
Training script for Audio 3D Gaussian Splatting
Adapted from AV-Cloud training pipeline for pure audio approach
"""

from __future__ import division, print_function, with_statement

import argparse
import os
import random
from importlib import import_module as impm

import _init_paths
import numpy as np
import torch
import torch.distributed as dist

from configs import cfg, update_config
from libs.utils import misc
from libs.utils.lr_scheduler import ExponentialLR
from libs.utils.utils import create_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Audio 3DGS Training')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/audio_3dgs.yaml',
        required=True,
        type=str)
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://10.5.38.36:23456',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        dest='rank',
        default=0,
        type=int,
        help='node rank for distributed training')
    
    # Additional arguments for overriding config
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Update config from file
    update_config(cfg, args)
        
    return args, cfg


def setup_distributed(args):
    """Setup distributed training"""
    if args.distributed:
        if args.local_rank != -1:
            args.gpu = args.local_rank
            torch.cuda.set_device(args.gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        else:
            print('Not using distributed mode')
            args.distributed = False


def main():
    args, cfg = parse_args()
    
    # Setup distributed training
    setup_distributed(args)
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Setup logger
    logger, final_output_dir = create_logger(cfg)
    logger.info(f"Using config: {args.yaml_file}")
    logger.info(f"Output directory: {final_output_dir}")
    
    # Set random seeds
    if cfg.get('seed', None):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Build dataset and data loader
    logger.info("Creating dataset...")
    dataset_module = impm(f"{cfg.dataset.name}")
    train_loader = dataset_module.make_data_loader(cfg, 'train', args.distributed)
    val_loader = dataset_module.make_data_loader(cfg, 'val', args.distributed)
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Build model
    logger.info("Creating model...")
    model_module = impm(f"libs.models.{cfg.model.file}")
    model = model_module.build_model(cfg)
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
        logger.info("Model moved to GPU")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler  
    from torch.optim.lr_scheduler import StepLR
    lr_scheduler = StepLR(optimizer, step_size=cfg.train.lr_decay_step, gamma=cfg.train.lr_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Build trainer
    trainer_module = impm(f"libs.trainers.{cfg.train.file}")
    trainer = trainer_module.build_trainer(
        cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger
    )
    
    # Set starting epoch
    trainer.epoch = start_epoch
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()