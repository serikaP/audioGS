"""
Test script for Audio 3D Gaussian Splatting
Adapted from AV-Cloud test pipeline
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from importlib import import_module as impm

import _init_paths
from configs import cfg, update_config
from libs.utils import misc
from libs.utils.utils import create_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Audio 3DGS Testing')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name',
        required=True,
        type=str)
    parser.add_argument(
        '--checkpoint',
        help='checkpoint path',
        required=True,
        type=str)
    parser.add_argument(
        '--output_dir',
        help='output directory for results',
        type=str)
    parser.add_argument(
        '--video',
        help='video/scene name',
        type=str)
    
    args, rest = parser.parse_known_args()
    
    # Update config
    update_config(cfg, args.yaml_file)
    cfg = misc.parse_command_line_and_update_config(cfg, rest)
    
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.video:
        cfg.dataset.video = args.video
        
    return args, cfg


def main():
    args, cfg = parse_args()
    
    # Setup logger
    logger = create_logger(cfg, cfg.output_dir, 'test')
    logger.info(f"Testing with config: {args.yaml_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Build dataset
    logger.info("Creating test dataset...")
    dataset_module = impm(f"libs.datasets.{cfg.dataset.dataset}")
    test_loader = dataset_module.make_data_loader(cfg, 'val', distributed=False)
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Build model
    logger.info("Creating model...")
    model_module = impm(f"libs.models.{cfg.model.file}")
    model = model_module.build_model(cfg)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    
    model.eval()
    
    # Build evaluator
    evaluator_module = impm("libs.evaluators.gen_eval")
    evaluator = evaluator_module.get_evaluator(cfg)
    
    # Test loop
    logger.info("Starting evaluation...")
    all_metrics = {}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            logger.info(f"Processing batch {i+1}/{len(test_loader)}")
            
            # Move to GPU
            cam_pose = batch['cam_pose'].cuda() if torch.cuda.is_available() else batch['cam_pose']
            source_audio = batch['source_audio'].cuda() if torch.cuda.is_available() else batch['source_audio']
            target_binaural = batch['target_binaural']
            
            # Forward pass
            pred_binaural = model(cam_pose, source_audio, is_val=True)
            
            # Move back to CPU for evaluation
            pred_binaural = pred_binaural.cpu()
            
            # Ensure same length
            min_len = min(pred_binaural.shape[-1], target_binaural.shape[-1])
            pred_binaural = pred_binaural[..., :min_len]
            target_binaural = target_binaural[..., :min_len]
            
            # Store for final evaluation
            all_predictions.append(pred_binaural.numpy())
            all_targets.append(target_binaural.numpy())
            
            # Evaluate metrics for this batch
            if evaluator:
                batch_metrics = evaluator.evaluate(
                    pred_binaural.numpy(),
                    target_binaural.numpy(),
                    cfg.dataset.sampling_rate
                )
                
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
    
    # Calculate average metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        avg_metrics[key] = np.mean(values)
        logger.info(f"{key}: {avg_metrics[key]:.4f}")
    
    # Save results
    results = {
        'metrics': avg_metrics,
        'predictions': np.concatenate(all_predictions, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'config': cfg
    }
    
    results_path = os.path.join(cfg.output_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for key, value in avg_metrics.items():
        logger.info(f"{key:>10}: {value:.4f}")
    logger.info("=" * 60)
    
    logger.info("Testing completed!")


if __name__ == '__main__':
    main()