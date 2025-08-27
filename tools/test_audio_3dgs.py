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
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Update config
    update_config(cfg, args)
    
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.video:
        cfg.dataset.video = args.video
        
    return args, cfg


def main():
    args, cfg = parse_args()
    
    # Setup logger
    logger, final_output_dir = create_logger(cfg)
    logger.info(f"Testing with config: {args.yaml_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {final_output_dir}")
    
    # Build dataset
    logger.info("Creating test dataset...")
    dataset_module = impm(f"{cfg.dataset.name}")
    test_loader = dataset_module.make_data_loader(cfg, 'val', distributed=False)
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Build model
    logger.info("Creating model...")
    model_module = impm(f"libs.models.{cfg.model.file}")
    model = model_module.build_model(cfg)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Successfully loaded checkpoint from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    
    model.eval()
    
    # Build evaluator
    from libs.evaluators.gen_eval import Evaluator
    evaluator = Evaluator(cfg, 'audio_3dgs', sampling_rate=cfg.dataset.sr)
    
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
                # Ensure correct format for evaluator: [2, length]
                pred_for_eval = pred_binaural.squeeze().numpy()
                target_for_eval = target_binaural.squeeze().numpy()
                
                # Ensure stereo format
                if pred_for_eval.ndim == 1:
                    pred_for_eval = np.stack([pred_for_eval, pred_for_eval])
                elif pred_for_eval.shape[0] != 2:
                    pred_for_eval = pred_for_eval.T
                    
                if target_for_eval.ndim == 1:
                    target_for_eval = np.stack([target_for_eval, target_for_eval])
                elif target_for_eval.shape[0] != 2:
                    target_for_eval = target_for_eval.T
                
                # Call evaluator (this adds metrics to evaluator.metrics internally)
                evaluator.evaluate(pred_for_eval, target_for_eval, sr=cfg.dataset.sr)
    
    # Get metrics from evaluator
    logger.info("Computing final metrics...")
    
    avg_metrics = {}
    for key, values in evaluator.metrics.items():
        if values:  # Only process non-empty metrics
            avg_metrics[key] = np.mean(values)
    
    # Map to standard metric names for reporting
    metric_mapping = {
        'stft_mse': 'MAG',      # Magnitude Spectrogram Distance
        'left_right_err': 'LRE', # Left-Right Energy Ratio Error  
        'env': 'ENV',           # Envelope Distance
        'dpam': 'RTE'           # RT60 Error (using DPAM as proxy)
    }
    
    final_metrics = {}
    for original_key, readable_key in metric_mapping.items():
        if original_key in avg_metrics:
            final_metrics[readable_key] = avg_metrics[original_key]
    
    # Save results
    results = {
        'metrics': avg_metrics,
        'predictions': np.concatenate(all_predictions, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'config': cfg
    }
    
    results_path = os.path.join(final_output_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("AUDIO 3DGS EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    # Print the four main metrics from AV-Cloud paper
    if final_metrics:
        for key, value in final_metrics.items():
            logger.info(f"{key:>10}: {value:.4f}")
    else:
        logger.info("No metrics computed - check evaluator functionality")
        
    # Also print all available metrics for debugging
    logger.info("-" * 60)
    logger.info("ALL COMPUTED METRICS:")
    for key, value in avg_metrics.items():
        logger.info(f"{key:>15}: {value:.4f}")
    logger.info("=" * 60)
    
    logger.info("Testing completed!")


if __name__ == '__main__':
    main()