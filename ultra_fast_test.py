#!/usr/bin/env python3
"""
Ultra-Fast Test Mode
Tests on different sized samples to quickly validate model
"""

import os
import sys
import time
import numpy as np
import torch
from landslide_model_improved import LandslideModelImproved
import rasterio
from rasterio.windows import Window


def ultra_fast_test(model_path, raster_paths, sample_size=1000):
    """
    Ultra-fast test on minimal sample
    
    Args:
        model_path: Path to trained model
        raster_paths: List of input raster paths
        sample_size: Number of pixels to test (default: 1000 for ~3 seconds)
    """
    
    print("\n" + "="*70)
    print(f"⚡ ULTRA-FAST TEST - {sample_size:,} pixels")
    print("="*70)
    
    start_time = time.time()
    
    # Load model
    print("Loading model...", end=' ', flush=True)
    model = LandslideModelImproved()
    model.load_model(model_path)
    print(f"✓ ({time.time()-start_time:.1f}s)")
    
    # Get reference raster for shape
    with rasterio.open(raster_paths[0]) as src:
        height, width = src.shape
        total_pixels = height * width
    
    # Calculate sample rate
    sample_rate = sample_size / total_pixels
    step = max(1, int(np.sqrt(1/sample_rate)))
    
    print(f"Raster size: {height} x {width} = {total_pixels:,} pixels")
    print(f"Sampling every {step} pixels...")
    
    # Read sampled data from all rasters
    sample_data = []
    valid_indices = None
    
    for i, raster_path in enumerate(raster_paths):
        with rasterio.open(raster_path) as src:
            # Read with stride to sample evenly
            data = src.read(1, out_shape=(height//step, width//step))
            flat_data = data.flatten()
            
            # Track valid pixels (first raster only)
            if valid_indices is None:
                nodata = src.nodata if src.nodata is not None else -9999
                valid_indices = (flat_data != nodata) & (~np.isnan(flat_data))
            
            sample_data.append(flat_data)
    
    # Stack and filter
    sample_data = np.column_stack(sample_data)
    sample_data = sample_data[valid_indices]
    
    actual_samples = len(sample_data)
    print(f"Valid samples: {actual_samples:,}")
    
    # If we have more samples than requested, subsample
    if actual_samples > sample_size:
        indices = np.random.choice(actual_samples, sample_size, replace=False)
        sample_data = sample_data[indices]
        actual_samples = sample_size
    
    print(f"\nPredicting on {actual_samples:,} pixels...", end=' ', flush=True)
    pred_start = time.time()
    
    # Convert to tensor and predict
    X_tensor = torch.FloatTensor(sample_data)
    if torch.cuda.is_available():
        X_tensor = X_tensor.cuda()
        model.model.cuda()
    
    with torch.no_grad():
        predictions = torch.sigmoid(model.model(X_tensor)).cpu().numpy().flatten()
    
    pred_time = time.time() - pred_start
    total_time = time.time() - start_time
    
    print(f"✓ ({pred_time:.1f}s)")
    
    # Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Prediction time: {pred_time:.1f}s")
    print(f"Speed: {actual_samples/pred_time:.0f} pixels/second")
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean:   {predictions.mean():.4f}")
    print(f"  Std:    {predictions.std():.4f}")
    print(f"  Min:    {predictions.min():.4f}")
    print(f"  Max:    {predictions.max():.4f}")
    print(f"  Median: {np.median(predictions):.4f}")
    
    # Distribution analysis
    print(f"\nDistribution:")
    bins = [0, 0.3, 0.7, 1.0]
    labels = ['Low Risk (0-0.3)', 'Moderate Risk (0.3-0.7)', 'High Risk (0.7-1.0)']
    
    for i in range(len(bins)-1):
        count = np.sum((predictions >= bins[i]) & (predictions < bins[i+1]))
        pct = 100 * count / len(predictions)
        print(f"  {labels[i]:25s}: {pct:5.1f}% ({count:,} pixels)")
    
    # Assessment
    std = predictions.std()
    mean = predictions.mean()
    
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if std < 0.10:
        print("⚠️  WARNING: Very low variance (Std < 0.10)")
        print("    Predictions are too similar - possible overfitting")
        print("    Consider retraining with simplified architecture")
    elif std < 0.15:
        print("⚠️  CAUTION: Low variance (Std < 0.15)")
        print("    Some binary behavior detected")
    elif std > 0.30:
        print("⚠️  CAUTION: High variance (Std > 0.30)")
        print("    Predictions may be too uncertain")
    else:
        print("✅ GOOD: Variance looks healthy (0.15 < Std < 0.30)")
    
    if mean < 0.3 or mean > 0.7:
        print("⚠️  WARNING: Mean far from 0.5 - model may be biased")
    else:
        print("✅ GOOD: Mean is balanced")
    
    print("="*70)
    
    # Estimate full prediction time
    pixels_per_sec = actual_samples / pred_time
    estimated_full_time = total_pixels / pixels_per_sec
    print(f"\nEstimated time for full raster: {estimated_full_time/60:.1f} minutes")
    
    return {
        'mean': predictions.mean(),
        'std': predictions.std(),
        'min': predictions.min(),
        'max': predictions.max(),
        'median': np.median(predictions),
        'time': total_time,
        'pred_time': pred_time,
        'samples': actual_samples,
        'speed': pixels_per_sec
    }


def main():
    """Run ultra-fast test"""
    
    # Configuration
    MODEL_PATH = "output4.pth"
    RASTERS_DIR = "test1"
    
    # Test sizes
    test_modes = {
        'tiny': 500,      # ~2 seconds
        'quick': 2000,    # ~5 seconds  
        'fast': 5000,     # ~10 seconds
        'standard': 10000 # ~20 seconds
    }
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in test_modes:
            sample_size = test_modes[mode]
        elif mode.isdigit():
            sample_size = int(mode)
        else:
            print(f"Unknown mode: {mode}")
            print(f"Available modes: {', '.join(test_modes.keys())}")
            print(f"Or specify a number of pixels")
            sys.exit(1)
    else:
        # Default to quick mode
        sample_size = test_modes['quick']
    
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[2]
    
    if len(sys.argv) > 3:
        RASTERS_DIR = sys.argv[3]
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(RASTERS_DIR):
        print(f"❌ Error: Rasters directory not found: {RASTERS_DIR}")
        sys.exit(1)
    
    # Get raster paths
    raster_paths = []
    for file in os.listdir(RASTERS_DIR):
        if file.endswith('.tif') or file.endswith('.tiff'):
            raster_paths.append(os.path.join(RASTERS_DIR, file))
    
    if not raster_paths:
        print(f"❌ Error: No .tif files found in {RASTERS_DIR}")
        sys.exit(1)
    
    print(f"Model: {MODEL_PATH}")
    print(f"Rasters: {len(raster_paths)} files from {RASTERS_DIR}")
    
    # Run test
    results = ultra_fast_test(MODEL_PATH, raster_paths, sample_size)
    
    return results


if __name__ == "__main__":
    print("\n" + "⚡"*35)
    print("ULTRA-FAST MODEL TEST")
    print("⚡"*35)
    print("\nUsage: python ultra_fast_test.py [mode] [model_path] [rasters_dir]")
    print("Modes: tiny (500px), quick (2k px), fast (5k px), standard (10k px)")
    print("Or specify exact number of pixels")
    print()
    
    main()
