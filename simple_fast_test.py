#!/usr/bin/env python3
"""
Simple Fast Test - Just run prediction on a small output size
Uses the same process_rasters method but outputs to a smaller file
"""

import os
import sys
import time
import numpy as np
import rasterio


def simple_fast_test(model_path, raster_dir, output_name="fast_test_output.tif"):
    """
    Run a fast test by processing all rasters but analyzing results quickly
    
    Args:
        model_path: Path to trained model (.pth file)
        raster_dir: Directory containing all aligned raster files
        output_name: Name for output file
    """
    
    print("\n" + "="*70)
    print("⚡ SIMPLE FAST TEST")
    print("="*70)
    
    # Import the model class
    from landslide_model_improved import LandslideModelImproved
    
    start_time = time.time()
    
    # Load model
    print("\n📦 Loading model...")
    model = LandslideModelImproved()
    model.load_model(model_path)
    load_time = time.time() - start_time
    print(f"   ✓ Loaded in {load_time:.1f}s")
    
    # Get all raster files
    print(f"\n📂 Scanning {raster_dir} for rasters...")
    all_files = [f for f in os.listdir(raster_dir) 
                 if f.endswith('.tif') and not f.endswith('.aux.xml')]
    
    # Find aligned rasters (prefer these)
    aligned_files = [f for f in all_files if 'aligned' in f.lower()]
    
    if len(aligned_files) >= 14:
        print(f"   ✓ Found {len(aligned_files)} aligned rasters")
        raster_files = aligned_files[:14]  # Take first 14
    else:
        print(f"   ✓ Found {len(all_files)} rasters")
        raster_files = all_files[:14]  # Take first 14
    
    # Create full paths
    raster_paths = [os.path.join(raster_dir, f) for f in raster_files]
    
    print(f"\n📊 Using {len(raster_paths)} rasters:")
    for i, path in enumerate(raster_paths, 1):
        print(f"   {i:2d}. {os.path.basename(path)}")
    
    # Get dimensions
    with rasterio.open(raster_paths[0]) as src:
        height, width = src.shape
        total_pixels = height * width
    
    print(f"\n📐 Raster size: {height} x {width} = {total_pixels:,} pixels")
    
    # Process
    output_path = os.path.join(os.path.dirname(model_path), output_name)
    print(f"\n🔮 Predicting...")
    print(f"   Output will be saved to: {output_path}")
    
    pred_start = time.time()
    
    try:
        model.process_rasters(raster_paths, output_path)
        pred_time = time.time() - pred_start
        total_time = time.time() - start_time
        
        print(f"\n   ✓ Prediction complete in {pred_time:.1f}s")
        
        # Analyze results
        print(f"\n📈 Analyzing results...")
        with rasterio.open(output_path) as src:
            pred_data = src.read(1)
        
        # Filter out nodata
        valid_mask = (pred_data != -9999) & (~np.isnan(pred_data))
        valid_preds = pred_data[valid_mask]
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"  Loading: {load_time:.1f}s")
        print(f"  Prediction: {pred_time/60:.1f} minutes")
        print(f"Speed: {total_pixels/pred_time:.0f} pixels/second")
        
        print(f"\nValid predictions: {len(valid_preds):,} / {total_pixels:,} ({100*len(valid_preds)/total_pixels:.1f}%)")
        
        if len(valid_preds) > 0:
            print(f"\nPrediction Statistics:")
            print(f"  Mean:   {valid_preds.mean():.4f}")
            print(f"  Std:    {valid_preds.std():.4f}")
            print(f"  Min:    {valid_preds.min():.4f}")
            print(f"  Max:    {valid_preds.max():.4f}")
            print(f"  Median: {np.median(valid_preds):.4f}")
            
            # Percentiles
            print(f"\nPercentiles:")
            for pct in [5, 25, 50, 75, 95]:
                val = np.percentile(valid_preds, pct)
                print(f"  {pct:2d}th: {val:.4f}")
            
            # Distribution analysis
            print(f"\nDistribution:")
            bins = [0, 0.3, 0.7, 1.0]
            labels = ['Low Risk (0-0.3)', 'Moderate Risk (0.3-0.7)', 'High Risk (0.7-1.0)']
            
            for i in range(len(bins)-1):
                count = np.sum((valid_preds >= bins[i]) & (valid_preds < bins[i+1]))
                pct = 100 * count / len(valid_preds)
                bar = '█' * int(pct / 2)
                print(f"  {labels[i]:25s}: {pct:5.1f}% {bar}")
            
            # Assessment
            std = valid_preds.std()
            mean = valid_preds.mean()
            
            print("\n" + "="*70)
            print("ASSESSMENT")
            print("="*70)
            
            issues = []
            good = []
            
            if std < 0.10:
                issues.append("⚠️  Very low variance (Std < 0.10) - predictions too similar")
                issues.append("    → Possible overfitting, consider simpler model")
            elif std < 0.15:
                issues.append("⚠️  Low variance (Std < 0.15) - some binary behavior")
            elif std > 0.30:
                issues.append("⚠️  High variance (Std > 0.30) - predictions may be too uncertain")
            else:
                good.append("✅ Variance is healthy (0.15 < Std < 0.30)")
            
            if mean < 0.3 or mean > 0.7:
                issues.append(f"⚠️  Mean far from 0.5 ({mean:.3f}) - model may be biased")
            else:
                good.append(f"✅ Mean is balanced ({mean:.3f})")
            
            # Check for spikes in distribution
            low_pct = 100 * np.sum(valid_preds < 0.3) / len(valid_preds)
            high_pct = 100 * np.sum(valid_preds > 0.7) / len(valid_preds)
            
            if low_pct > 70 or high_pct > 70:
                issues.append(f"⚠️  Distribution too skewed (Low:{low_pct:.1f}% High:{high_pct:.1f}%)")
            else:
                good.append("✅ Distribution is reasonable")
            
            # Print assessment
            if good:
                for msg in good:
                    print(msg)
            
            if issues:
                print()
                for msg in issues:
                    print(msg)
            
            if not issues:
                print("✅ Model predictions look GOOD!")
            
            print("="*70)
            print(f"\n💾 Output saved to: {output_path}")
            print(f"   Open in QGIS to visually inspect the predictions")
            
            return {
                'mean': float(valid_preds.mean()),
                'std': float(valid_preds.std()),
                'min': float(valid_preds.min()),
                'max': float(valid_preds.max()),
                'median': float(np.median(valid_preds)),
                'time': total_time,
                'pred_time': pred_time,
                'samples': len(valid_preds),
                'speed': total_pixels/pred_time,
                'output_path': output_path
            }
        else:
            print("❌ No valid predictions found!")
            return None
            
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run simple fast test"""
    
    # Default paths
    MODEL_PATH = "/home/anees/OneDrive/geoProject/Durban/output5.pth"
    RASTERS_DIR = "/home/anees/OneDrive/geoProject/Durban/finalRasters"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    
    if len(sys.argv) > 2:
        RASTERS_DIR = sys.argv[2]
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        print(f"\nUsage: python {sys.argv[0]} [model_path] [rasters_dir]")
        sys.exit(1)
    
    if not os.path.exists(RASTERS_DIR):
        print(f"❌ Error: Rasters directory not found: {RASTERS_DIR}")
        sys.exit(1)
    
    print(f"✓ Model: {MODEL_PATH}")
    print(f"✓ Rasters directory: {RASTERS_DIR}")
    
    # Run test
    results = simple_fast_test(MODEL_PATH, RASTERS_DIR)
    
    if results:
        print("\n" + "="*70)
        print("TEST COMPLETE!")
        print("="*70)
        print(f"\nTime: {results['time']/60:.1f} minutes")
        print(f"Speed: {results['speed']:.0f} pixels/second")
        print(f"Output: {results['output_path']}")
    
    return results


if __name__ == "__main__":
    print("\n" + "⚡"*35)
    print("SIMPLE FAST MODEL TEST")
    print("⚡"*35)
    print("\nThis will run a full prediction and analyze the results.")
    print("Usage: python simple_fast_test.py [model_path] [rasters_dir]")
    print()
    
    main()
