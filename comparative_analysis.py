#!/usr/bin/env python3
"""
Comparative Analysis: Before vs After Phase 1 Improvements
Analyzes the performance gains from implementing the critical fixes
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings('ignore')

# Try to import geospatial dependencies
try:
    import rasterio
    import geopandas as gpd
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False

def comparative_analysis():
    """Compare before and after improvements"""
    
    print("=" * 80)
    print("PHASE 1 IMPROVEMENTS - COMPARATIVE ANALYSIS")
    print("=" * 80)
    print("🔍 Comparing BEFORE (output.pth, map) vs AFTER (output2.pth, map2)")
    print("=" * 80)
    
    # File paths
    before_model = "/home/anees/Projects/annlandslide_train/outputs/output.pth"
    after_model = "/home/anees/Projects/annlandslide_train/outputs/output2.pth"
    before_map = "/home/anees/Projects/annlandslide_train/outputs/map"
    after_map = "/home/anees/Projects/annlandslide_train/outputs/map2"
    landslide_points = "/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/DurbanRasters/clipped_landslidePoints_lo19.gpkg"
    
    # Check file existence
    files_exist = True
    for filepath, name in [(before_model, "Before Model"), (after_model, "After Model"),
                          (before_map, "Before Map"), (after_map, "After Map")]:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✅ {name}: {os.path.basename(filepath)} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: {os.path.basename(filepath)} (NOT FOUND)")
            files_exist = False
    
    if not files_exist:
        print("\n❌ Missing required files. Please ensure both models and maps exist.")
        return
    
    print("\n" + "="*60)
    
    # 1. Compare Model Architecture and Training Metrics
    print("1️⃣  MODEL COMPARISON")
    print("-" * 30)
    compare_models(before_model, after_model)
    
    # 2. Compare Susceptibility Maps
    if GEOSPATIAL_AVAILABLE:
        print("\n2️⃣  SUSCEPTIBILITY MAP COMPARISON")
        print("-" * 40)
        compare_susceptibility_maps(before_map, after_map)
        
        # 3. Compare Performance at Landslide Locations
        print("\n3️⃣  LANDSLIDE LOCATION PERFORMANCE")
        print("-" * 40)
        compare_landslide_performance(before_map, after_map, landslide_points)
    else:
        print("\n⚠️  Geospatial analysis skipped - missing dependencies")
    
    # 4. Generate Improvement Summary
    print("\n4️⃣  IMPROVEMENT SUMMARY")
    print("-" * 30)
    generate_improvement_summary()

def compare_models(before_path, after_path):
    """Compare model architectures and training metrics"""
    
    print("📊 LOADING MODELS...")
    
    # Load models
    before_data = torch.load(before_path, map_location='cpu', weights_only=False)
    after_data = torch.load(after_path, map_location='cpu', weights_only=False)
    
    print(f"✅ Before model: {os.path.getsize(before_path)/1024:.1f} KB")
    print(f"✅ After model:  {os.path.getsize(after_path)/1024:.1f} KB")
    
    # Compare model components
    print(f"\n🧠 MODEL COMPONENTS:")
    print(f"   Before: {list(before_data.keys())}")
    print(f"   After:  {list(after_data.keys())}")
    
    # Compare training metrics
    if 'training_info' in before_data and 'training_info' in after_data:
        compare_training_metrics(before_data['training_info'], after_data['training_info'])
    else:
        print("⚠️  Training info not available for comparison")
    
    # Compare model architecture
    if 'model_state_dict' in before_data and 'model_state_dict' in after_data:
        compare_model_architecture(before_data['model_state_dict'], after_data['model_state_dict'])

def compare_training_metrics(before_info, after_info):
    """Compare training performance metrics"""
    
    print(f"\n📈 TRAINING PERFORMANCE COMPARISON:")
    
    # Key metrics to compare
    metrics = ['accuracy', 'best_f1', 'precision', 'recall', 'auc_roc']
    
    improvements = {}
    
    print(f"{'Metric':<12} {'Before':<8} {'After':<8} {'Change':<10} {'Improvement'}")
    print("-" * 55)
    
    for metric in metrics:
        if metric in before_info and metric in after_info:
            before_val = before_info[metric]
            after_val = after_info[metric]
            change = after_val - before_val
            improvement = (change / before_val * 100) if before_val != 0 else 0
            
            improvements[metric] = improvement
            
            status = "✅" if change > 0 else "❌" if change < 0 else "➖"
            print(f"{metric:<12} {before_val:<8.4f} {after_val:<8.4f} {change:+8.4f} {status} {improvement:+6.1f}%")
    
    # Training convergence comparison
    if 'train_losses' in before_info and 'train_losses' in after_info:
        compare_training_convergence(before_info, after_info)
    
    return improvements

def compare_training_convergence(before_info, after_info):
    """Compare training convergence patterns"""
    
    print(f"\n📊 TRAINING CONVERGENCE:")
    
    before_train = before_info.get('train_losses', [])
    before_val = before_info.get('val_losses', [])
    after_train = after_info.get('train_losses', [])
    after_val = after_info.get('val_losses', [])
    
    if before_train and before_val and after_train and after_val:
        # Calculate overfitting indicators
        before_gap = before_val[-1] - before_train[-1] if before_val and before_train else 0
        after_gap = after_val[-1] - after_train[-1] if after_val and after_train else 0
        
        print(f"   Epochs - Before: {len(before_train):3d}, After: {len(after_train):3d}")
        print(f"   Final Train Loss - Before: {before_train[-1]:.6f}, After: {after_train[-1]:.6f}")
        print(f"   Final Val Loss   - Before: {before_val[-1]:.6f}, After: {after_val[-1]:.6f}")
        print(f"   Overfitting Gap  - Before: {before_gap:+.6f}, After: {after_gap:+.6f}")
        
        gap_improvement = ((before_gap - after_gap) / abs(before_gap) * 100) if before_gap != 0 else 0
        status = "✅" if after_gap < before_gap else "❌"
        print(f"   Gap Improvement: {status} {gap_improvement:+.1f}%")
        
        # Create training curves comparison
        create_training_comparison_plots(before_info, after_info)

def compare_model_architecture(before_state, after_state):
    """Compare model architectures"""
    
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    
    # Count parameters
    before_params = sum(p.numel() for p in before_state.values())
    after_params = sum(p.numel() for p in after_state.values())
    
    print(f"   Parameters - Before: {before_params:,}, After: {after_params:,}")
    
    # Compare layer structures
    before_layers = [k for k in before_state.keys() if 'weight' in k]
    after_layers = [k for k in after_state.keys() if 'weight' in k]
    
    print(f"   Layers - Before: {len(before_layers)}, After: {len(after_layers)}")

def compare_susceptibility_maps(before_map_path, after_map_path):
    """Compare susceptibility map statistics"""
    
    try:
        print("📊 LOADING SUSCEPTIBILITY MAPS...")
        
        with rasterio.open(before_map_path) as before_src:
            before_data = before_src.read(1)
            before_transform = before_src.transform
            before_crs = before_src.crs
            
        with rasterio.open(after_map_path) as after_src:
            after_data = after_src.read(1)
            after_transform = after_src.transform
            after_crs = after_src.crs
        
        # Filter valid data
        before_valid = before_data[~np.isnan(before_data)]
        after_valid = after_data[~np.isnan(after_data)]
        
        print(f"✅ Before map: {before_data.shape}, {len(before_valid):,} valid pixels")
        print(f"✅ After map:  {after_data.shape}, {len(after_valid):,} valid pixels")
        
        # Compare statistics
        print(f"\n📈 SUSCEPTIBILITY STATISTICS:")
        
        stats_comparison = [
            ('Mean', np.mean(before_valid), np.mean(after_valid)),
            ('Std', np.std(before_valid), np.std(after_valid)),
            ('Min', np.min(before_valid), np.min(after_valid)),
            ('Max', np.max(before_valid), np.max(after_valid)),
            ('Median', np.median(before_valid), np.median(after_valid))
        ]
        
        print(f"{'Statistic':<10} {'Before':<8} {'After':<8} {'Change':<10}")
        print("-" * 40)
        
        for stat_name, before_val, after_val in stats_comparison:
            change = after_val - before_val
            print(f"{stat_name:<10} {before_val:<8.4f} {after_val:<8.4f} {change:+8.4f}")
        
        # Risk level distribution comparison
        compare_risk_distributions(before_valid, after_valid)
        
        # Create map comparison plots
        create_map_comparison_plots(before_valid, after_valid)
        
        return before_data, after_data, before_transform, after_transform
        
    except Exception as e:
        print(f"❌ Error comparing maps: {e}")
        return None, None, None, None

def compare_risk_distributions(before_data, after_data):
    """Compare risk level distributions"""
    
    print(f"\n🎯 RISK LEVEL DISTRIBUTION:")
    
    risk_levels = [
        ('Very Low', 0.0, 0.2),
        ('Low', 0.2, 0.4),
        ('Moderate', 0.4, 0.6),
        ('High', 0.6, 0.8),
        ('Very High', 0.8, 1.0)
    ]
    
    print(f"{'Risk Level':<12} {'Before %':<10} {'After %':<10} {'Change'}")
    print("-" * 45)
    
    for level_name, min_val, max_val in risk_levels:
        before_count = ((before_data >= min_val) & (before_data <= max_val)).sum()
        after_count = ((after_data >= min_val) & (after_data <= max_val)).sum()
        
        before_pct = before_count / len(before_data) * 100
        after_pct = after_count / len(after_data) * 100
        change = after_pct - before_pct
        
        print(f"{level_name:<12} {before_pct:<10.1f} {after_pct:<10.1f} {change:+7.1f}%")

def compare_landslide_performance(before_map_path, after_map_path, landslide_points_path):
    """Compare performance at actual landslide locations"""
    
    try:
        print("📍 ANALYZING LANDSLIDE LOCATION PERFORMANCE...")
        
        # Load landslide points
        landslides = gpd.read_file(landslide_points_path)
        print(f"✅ Loaded {len(landslides)} landslide points")
        
        # Extract susceptibility values at landslide locations
        before_values = extract_landslide_values(before_map_path, landslides)
        after_values = extract_landslide_values(after_map_path, landslides)
        
        if before_values is not None and after_values is not None:
            print(f"✅ Extracted values for {len(before_values)} landslide locations")
            
            # Compare statistics at landslide locations
            print(f"\n📊 SUSCEPTIBILITY AT LANDSLIDE LOCATIONS:")
            
            landslide_stats = [
                ('Mean', np.mean(before_values), np.mean(after_values)),
                ('Std', np.std(before_values), np.std(after_values)),
                ('Min', np.min(before_values), np.min(after_values)),
                ('Max', np.max(before_values), np.max(after_values)),
                ('Median', np.median(before_values), np.median(after_values))
            ]
            
            print(f"{'Statistic':<10} {'Before':<8} {'After':<8} {'Change'}")
            print("-" * 35)
            
            for stat_name, before_val, after_val in landslide_stats:
                change = after_val - before_val
                print(f"{stat_name:<10} {before_val:<8.4f} {after_val:<8.4f} {change:+8.4f}")
            
            # Compare risk level capture
            compare_landslide_risk_capture(before_values, after_values)
            
            # Create landslide performance plots
            create_landslide_comparison_plots(before_values, after_values)
        
    except Exception as e:
        print(f"❌ Error analyzing landslide performance: {e}")

def extract_landslide_values(raster_path, landslides):
    """Extract raster values at landslide point locations"""
    
    try:
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            
            susceptibility_values = []
            
            for idx, point in landslides.iterrows():
                try:
                    geom = point.geometry
                    x, y = geom.x, geom.y
                    
                    # Convert to pixel coordinates
                    col, row = ~transform * (x, y)
                    col, row = int(col), int(row)
                    
                    # Check bounds and extract value
                    if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
                        value = raster_data[row, col]
                        if not np.isnan(value) and value != src.nodata:
                            susceptibility_values.append(value)
                
                except Exception:
                    continue
            
            return np.array(susceptibility_values)
            
    except Exception:
        return None

def compare_landslide_risk_capture(before_values, after_values):
    """Compare landslide capture rates by risk level"""
    
    print(f"\n🎯 LANDSLIDE CAPTURE BY RISK LEVEL:")
    
    risk_thresholds = [
        ('Very Low', 0.0, 0.2),
        ('Low', 0.2, 0.4), 
        ('Moderate', 0.4, 0.6),
        ('High', 0.6, 0.8),
        ('Very High', 0.8, 1.0)
    ]
    
    print(f"{'Risk Level':<12} {'Before':<8} {'After':<8} {'Change'}")
    print("-" * 40)
    
    total_landslides = len(before_values)
    
    for level_name, min_val, max_val in risk_thresholds:
        before_count = ((before_values >= min_val) & (before_values <= max_val)).sum()
        after_count = ((after_values >= min_val) & (after_values <= max_val)).sum()
        
        before_pct = before_count / total_landslides * 100
        after_pct = after_count / total_landslides * 100
        change = after_pct - before_pct
        
        status = "✅" if change > 0 else "❌" if change < 0 else "➖"
        print(f"{level_name:<12} {before_pct:<7.1f}% {after_pct:<7.1f}% {status} {change:+5.1f}%")
    
    # Key performance indicators
    before_high_plus = ((before_values >= 0.6) & (before_values <= 1.0)).sum()
    after_high_plus = ((after_values >= 0.6) & (after_values <= 1.0)).sum()
    
    before_mod_plus = ((before_values >= 0.4) & (before_values <= 1.0)).sum()
    after_mod_plus = ((after_values >= 0.4) & (after_values <= 1.0)).sum()
    
    print(f"\n📈 KEY PERFORMANCE INDICATORS:")
    print(f"   High+ Risk Capture:     {before_high_plus/total_landslides*100:.1f}% → {after_high_plus/total_landslides*100:.1f}% ({(after_high_plus-before_high_plus)/total_landslides*100:+.1f}%)")
    print(f"   Moderate+ Risk Capture: {before_mod_plus/total_landslides*100:.1f}% → {after_mod_plus/total_landslides*100:.1f}% ({(after_mod_plus-before_mod_plus)/total_landslides*100:+.1f}%)")

def create_training_comparison_plots(before_info, after_info):
    """Create training comparison visualizations"""
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Convergence Comparison: Before vs After Improvements', fontsize=14, fontweight='bold')
        
        # Training and validation losses
        before_train = before_info.get('train_losses', [])
        before_val = before_info.get('val_losses', [])
        after_train = after_info.get('train_losses', [])
        after_val = after_info.get('val_losses', [])
        
        if before_train and after_train:
            # Loss curves
            epochs_before = range(1, len(before_train) + 1)
            epochs_after = range(1, len(after_train) + 1)
            
            axes[0].plot(epochs_before, before_train, 'b-', label='Before - Train', linewidth=2, alpha=0.7)
            axes[0].plot(epochs_before, before_val, 'r-', label='Before - Val', linewidth=2, alpha=0.7)
            axes[0].plot(epochs_after, after_train, 'b--', label='After - Train', linewidth=2)
            axes[0].plot(epochs_after, after_val, 'r--', label='After - Val', linewidth=2)
            
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Overfitting gap comparison
            before_gap = np.array(before_val) - np.array(before_train) if len(before_val) == len(before_train) else []
            after_gap = np.array(after_val) - np.array(after_train) if len(after_val) == len(after_train) else []
            
            if len(before_gap) > 0 and len(after_gap) > 0:
                axes[1].plot(epochs_before, before_gap, 'purple', linewidth=2, alpha=0.7, label='Before')
                axes[1].plot(epochs_after, after_gap, 'orange', linewidth=2, label='After')
                axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Validation - Training Loss')
                axes[1].set_title('Overfitting Gap (Lower is Better)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_improvement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training comparison plots saved: training_improvement_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Could not create training plots: {e}")

def create_map_comparison_plots(before_data, after_data):
    """Create susceptibility map comparison plots"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Susceptibility Map Comparison: Before vs After Improvements', fontsize=16, fontweight='bold')
        
        # Distribution histograms
        axes[0, 0].hist(before_data, bins=50, alpha=0.7, color='blue', density=True, label='Before')
        axes[0, 0].hist(after_data, bins=50, alpha=0.7, color='red', density=True, label='After')
        axes[0, 0].set_xlabel('Susceptibility Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Susceptibility Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        box_data = [before_data, after_data]
        box_labels = ['Before\nImprovements', 'After\nImprovements']
        axes[0, 1].boxplot(box_data, labels=box_labels)
        axes[0, 1].set_ylabel('Susceptibility Value')
        axes[0, 1].set_title('Distribution Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Risk level pie charts
        risk_levels = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        # Before pie chart
        before_counts = []
        for min_val, max_val in risk_levels:
            count = ((before_data >= min_val) & (before_data <= max_val)).sum()
            before_counts.append(count)
        
        axes[1, 0].pie(before_counts, labels=risk_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Before: Risk Level Distribution')
        
        # After pie chart
        after_counts = []
        for min_val, max_val in risk_levels:
            count = ((after_data >= min_val) & (after_data <= max_val)).sum()
            after_counts.append(count)
        
        axes[1, 1].pie(after_counts, labels=risk_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('After: Risk Level Distribution')
        
        plt.tight_layout()
        plt.savefig('susceptibility_map_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Map comparison plots saved: susceptibility_map_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Could not create map comparison plots: {e}")

def create_landslide_comparison_plots(before_values, after_values):
    """Create landslide performance comparison plots"""
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Landslide Prediction Performance: Before vs After Improvements', fontsize=14, fontweight='bold')
        
        # Susceptibility at landslide locations
        axes[0].hist(before_values, bins=20, alpha=0.7, color='blue', density=True, label='Before')
        axes[0].hist(after_values, bins=20, alpha=0.7, color='red', density=True, label='After')
        axes[0].set_xlabel('Susceptibility Value at Landslide Locations')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Susceptibility at Landslide Locations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        box_data = [before_values, after_values]
        box_labels = ['Before', 'After']
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('Susceptibility Value')
        axes[1].set_title('Landslide Location Predictions')
        axes[1].grid(True, alpha=0.3)
        
        # Risk level capture comparison
        risk_levels = ['Very Low\n(≤0.2)', 'Low\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 'High\n(0.6-0.8)', 'Very High\n(>0.8)']
        risk_thresholds = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        
        before_counts = []
        after_counts = []
        
        for min_val, max_val in risk_thresholds:
            before_count = ((before_values >= min_val) & (before_values <= max_val)).sum()
            after_count = ((after_values >= min_val) & (after_values <= max_val)).sum()
            before_counts.append(before_count / len(before_values) * 100)
            after_counts.append(after_count / len(after_values) * 100)
        
        x = np.arange(len(risk_levels))
        width = 0.35
        
        bars1 = axes[2].bar(x - width/2, before_counts, width, label='Before', alpha=0.7, color='blue')
        bars2 = axes[2].bar(x + width/2, after_counts, width, label='After', alpha=0.7, color='red')
        
        axes[2].set_xlabel('Risk Level')
        axes[2].set_ylabel('Percentage of Landslides (%)')
        axes[2].set_title('Landslide Capture by Risk Level')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(risk_levels)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[2].annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('landslide_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Landslide performance plots saved: landslide_performance_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Could not create landslide performance plots: {e}")

def generate_improvement_summary():
    """Generate overall improvement summary"""
    
    print("🎯 PHASE 1 IMPROVEMENTS ANALYSIS COMPLETE!")
    print("\n🔍 KEY FINDINGS:")
    print("   • Training convergence and stability improvements")
    print("   • Changes in susceptibility map distribution")
    print("   • Performance at actual landslide locations")
    print("   • Risk level capture improvements")
    
    print("\n📊 GENERATED ANALYSIS FILES:")
    print("   • training_improvement_comparison.png - Training curves analysis")
    print("   • susceptibility_map_comparison.png - Map distribution changes")
    print("   • landslide_performance_comparison.png - Landslide prediction performance")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Review the generated comparison plots")
    print("   2. Analyze performance improvements in key metrics")
    print("   3. Consider Phase 2 improvements if needed")
    print("   4. Deploy improved model for operational use")

if __name__ == "__main__":
    if not GEOSPATIAL_AVAILABLE:
        print("Installing geospatial dependencies...")
        os.system("pip install rasterio geopandas")
        print("Dependencies installed. Please re-run the script.")
        sys.exit(1)
    
    comparative_analysis()