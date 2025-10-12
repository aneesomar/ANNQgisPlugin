#!/usr/bin/env python3
"""
Comprehensive Analysis of ANN Landslide Susceptibility Plugin Performance
Analyzes prediction accuracy against known landslide occurrences and model characteristics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import GDAL/rasterio for raster analysis
try:
    import rasterio
    from rasterio.plot import show
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Installing...")
    os.system("pip install rasterio")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Installing...")
    os.system("pip install geopandas")

def analyze_susceptibility_performance():
    """Main analysis function"""
    
    print("=" * 80)
    print("ANN LANDSLIDE SUSCEPTIBILITY PLUGIN - PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Paths
    susceptibility_raster = "/home/anees/Projects/annlandslide_train/outputs/map"
    landslide_points = "/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/DurbanRasters/clipped_landslidePoints_lo19.gpkg"
    model_file = "/home/anees/Projects/annlandslide_train/outputs/output.pth"
    
    # Check file existence
    if not os.path.exists(susceptibility_raster):
        print(f"❌ Susceptibility raster not found: {susceptibility_raster}")
        return
    
    if not os.path.exists(landslide_points):
        print(f"❌ Landslide points not found: {landslide_points}")
        return
        
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    print(f"✅ Found susceptibility raster: {os.path.basename(susceptibility_raster)}")
    print(f"✅ Found landslide points: {os.path.basename(landslide_points)}")
    print(f"✅ Found model file: {os.path.basename(model_file)}")
    print()
    
    # 1. Analyze Susceptibility Map Statistics
    print("1️⃣  SUSCEPTIBILITY MAP ANALYSIS")
    print("-" * 50)
    analyze_susceptibility_map(susceptibility_raster)
    
    # 2. Analyze Model Performance at Landslide Locations
    print("\n2️⃣  LANDSLIDE LOCATION PERFORMANCE")
    print("-" * 50)
    analyze_landslide_performance(susceptibility_raster, landslide_points)
    
    # 3. Analyze Trained Model
    print("\n3️⃣  MODEL ARCHITECTURE ANALYSIS")
    print("-" * 50)
    analyze_trained_model(model_file)
    
    # 4. Generate Improvement Recommendations
    print("\n4️⃣  IMPROVEMENT RECOMMENDATIONS")
    print("-" * 50)
    generate_recommendations()

def analyze_susceptibility_map(raster_path):
    """Analyze the generated susceptibility map statistics"""
    
    try:
        if RASTERIO_AVAILABLE:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                # Handle nodata
                nodata = src.nodata
                if nodata is not None:
                    valid_data = data[data != nodata]
                else:
                    valid_data = data[~np.isnan(data)]
                
                # Basic statistics
                print(f"📊 Raster dimensions: {data.shape}")
                print(f"📊 Valid pixels: {len(valid_data):,}")
                print(f"📊 NoData pixels: {len(data) - len(valid_data):,}")
                print()
                
                # Susceptibility statistics
                print("📈 SUSCEPTIBILITY STATISTICS:")
                print(f"   Mean: {valid_data.mean():.4f}")
                print(f"   Std:  {valid_data.std():.4f}")
                print(f"   Min:  {valid_data.min():.4f}")
                print(f"   Max:  {valid_data.max():.4f}")
                print()
                
                # Percentiles
                percentiles = [5, 10, 25, 50, 75, 90, 95]
                print("📊 PERCENTILES:")
                for p in percentiles:
                    val = np.percentile(valid_data, p)
                    print(f"   {p:2d}th: {val:.4f}")
                print()
                
                # Risk classification
                classify_risk_levels(valid_data)
                
                # Create visualization
                create_susceptibility_plots(valid_data, "susceptibility_analysis")
                
        else:
            print("❌ Cannot analyze raster - rasterio not available")
            
    except Exception as e:
        print(f"❌ Error analyzing susceptibility map: {e}")

def classify_risk_levels(data):
    """Classify pixels into risk levels"""
    
    # Standard risk classification thresholds
    very_low = (data <= 0.2).sum()
    low = ((data > 0.2) & (data <= 0.4)).sum()
    moderate = ((data > 0.4) & (data <= 0.6)).sum()
    high = ((data > 0.6) & (data <= 0.8)).sum()
    very_high = (data > 0.8).sum()
    
    total = len(data)
    
    print("🎯 RISK CLASSIFICATION:")
    print(f"   Very Low  (≤0.2): {very_low:7,} pixels ({very_low/total*100:5.1f}%)")
    print(f"   Low    (0.2-0.4): {low:7,} pixels ({low/total*100:5.1f}%)")
    print(f"   Moderate (0.4-0.6): {moderate:7,} pixels ({moderate/total*100:5.1f}%)")
    print(f"   High   (0.6-0.8): {high:7,} pixels ({high/total*100:5.1f}%)")
    print(f"   Very High (>0.8): {very_high:7,} pixels ({very_high/total*100:5.1f}%)")
    print()

def analyze_landslide_performance(raster_path, points_path):
    """Analyze how well predictions match actual landslide locations"""
    
    try:
        if not (RASTERIO_AVAILABLE and GEOPANDAS_AVAILABLE):
            print("❌ Cannot analyze landslide performance - missing dependencies")
            return
            
        # Load landslide points
        landslides = gpd.read_file(points_path)
        print(f"📍 Loaded {len(landslides)} landslide points")
        
        # Load susceptibility raster
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Extract susceptibility values at landslide locations
            susceptibility_values = []
            
            for idx, point in landslides.iterrows():
                try:
                    # Get pixel coordinates
                    geom = point.geometry
                    x, y = geom.x, geom.y
                    
                    # Convert to pixel coordinates
                    col, row = ~transform * (x, y)
                    col, row = int(col), int(row)
                    
                    # Check bounds
                    if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
                        value = raster_data[row, col]
                        if not np.isnan(value) and value != src.nodata:
                            susceptibility_values.append(value)
                    
                except Exception as e:
                    continue
            
            susceptibility_values = np.array(susceptibility_values)
            
            if len(susceptibility_values) > 0:
                analyze_landslide_susceptibility_values(susceptibility_values, raster_data)
            else:
                print("❌ No valid susceptibility values extracted at landslide locations")
                
    except Exception as e:
        print(f"❌ Error analyzing landslide performance: {e}")

def analyze_landslide_susceptibility_values(landslide_values, full_raster):
    """Analyze susceptibility values at landslide locations vs general population"""
    
    # Remove nodata from full raster
    valid_raster = full_raster[~np.isnan(full_raster)]
    if hasattr(full_raster, 'mask'):
        valid_raster = full_raster[~full_raster.mask]
    
    print(f"📊 Extracted susceptibility values for {len(landslide_values)} landslide locations")
    print()
    
    # Statistics at landslide locations
    print("🎯 SUSCEPTIBILITY AT LANDSLIDE LOCATIONS:")
    print(f"   Mean: {landslide_values.mean():.4f}")
    print(f"   Std:  {landslide_values.std():.4f}")
    print(f"   Min:  {landslide_values.min():.4f}")
    print(f"   Max:  {landslide_values.max():.4f}")
    print()
    
    # Compare with overall population
    print("📈 COMPARISON WITH FULL RASTER:")
    print(f"   Full raster mean: {valid_raster.mean():.4f}")
    print(f"   Landslide mean:   {landslide_values.mean():.4f}")
    print(f"   Difference:       {landslide_values.mean() - valid_raster.mean():.4f}")
    print()
    
    # Statistical test
    try:
        t_stat, p_value = stats.ttest_ind(landslide_values, 
                                         np.random.choice(valid_raster, min(len(landslide_values)*10, len(valid_raster))))
        print(f"📊 T-test (landslides vs random sample):")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value:     {p_value:.2e}")
        print(f"   Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")
        print()
    except Exception as e:
        print(f"⚠️  Could not perform statistical test: {e}")
    
    # Risk level performance
    analyze_risk_level_performance(landslide_values)
    
    # Create performance plots
    create_performance_plots(landslide_values, valid_raster)

def analyze_risk_level_performance(landslide_values):
    """Analyze what risk levels the landslides fall into"""
    
    very_low = (landslide_values <= 0.2).sum()
    low = ((landslide_values > 0.2) & (landslide_values <= 0.4)).sum()
    moderate = ((landslide_values > 0.4) & (landslide_values <= 0.6)).sum()
    high = ((landslide_values > 0.6) & (landslide_values <= 0.8)).sum()
    very_high = (landslide_values > 0.8).sum()
    
    total = len(landslide_values)
    
    print("🎯 LANDSLIDES BY PREDICTED RISK LEVEL:")
    print(f"   Very Low  (≤0.2): {very_low:3d} landslides ({very_low/total*100:5.1f}%)")
    print(f"   Low    (0.2-0.4): {low:3d} landslides ({low/total*100:5.1f}%)")
    print(f"   Moderate (0.4-0.6): {moderate:3d} landslides ({moderate/total*100:5.1f}%)")
    print(f"   High   (0.6-0.8): {high:3d} landslides ({high/total*100:5.1f}%)")
    print(f"   Very High (>0.8): {very_high:3d} landslides ({very_high/total*100:5.1f}%)")
    print()
    
    # Performance assessment
    high_risk_landslides = high + very_high
    moderate_plus_landslides = moderate + high + very_high
    
    print("📊 PERFORMANCE METRICS:")
    print(f"   Landslides in High/Very High risk: {high_risk_landslides}/{total} ({high_risk_landslides/total*100:.1f}%)")
    print(f"   Landslides in Moderate+ risk:     {moderate_plus_landslides}/{total} ({moderate_plus_landslides/total*100:.1f}%)")
    print()

def analyze_trained_model(model_path):
    """Analyze the trained model architecture and parameters"""
    
    try:
        # Load model
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"📋 Model file size: {os.path.getsize(model_path)/1024:.1f} KB")
        
        # Analyze model components
        print("\n🧠 MODEL COMPONENTS:")
        for key in model_data.keys():
            if isinstance(model_data[key], dict):
                print(f"   {key}: {len(model_data[key])} items")
            elif isinstance(model_data[key], torch.nn.Module):
                print(f"   {key}: PyTorch model")
            elif hasattr(model_data[key], '__len__'):
                print(f"   {key}: {len(model_data[key])} items")
            else:
                print(f"   {key}: {type(model_data[key]).__name__}")
        
        # Analyze model architecture if available
        if 'model_state_dict' in model_data:
            analyze_model_architecture(model_data['model_state_dict'])
        elif 'state_dict' in model_data:
            analyze_model_architecture(model_data['state_dict'])
        
        # Analyze training info if available
        if 'training_info' in model_data:
            analyze_training_info(model_data['training_info'])
        
        # Analyze feature selection if available
        if 'selected_features' in model_data:
            analyze_feature_selection(model_data['selected_features'])
            
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")

def analyze_model_architecture(state_dict):
    """Analyze neural network architecture from state dict"""
    
    print("\n🏗️  MODEL ARCHITECTURE:")
    
    # Count parameters by layer
    layer_info = {}
    total_params = 0
    
    for name, param in state_dict.items():
        layer_name = name.split('.')[0]  # Get base layer name
        param_count = param.numel()
        total_params += param_count
        
        if layer_name not in layer_info:
            layer_info[layer_name] = {'params': 0, 'tensors': []}
        
        layer_info[layer_name]['params'] += param_count
        layer_info[layer_name]['tensors'].append(f"{name}: {list(param.shape)}")
    
    print(f"   Total parameters: {total_params:,}")
    print()
    
    for layer, info in layer_info.items():
        print(f"   {layer}:")
        print(f"     Parameters: {info['params']:,}")
        for tensor_info in info['tensors']:
            print(f"     {tensor_info}")
    
    # Estimate model complexity
    if total_params < 10000:
        complexity = "Simple"
    elif total_params < 100000:
        complexity = "Moderate"
    else:
        complexity = "Complex"
    
    print(f"\n   Model complexity: {complexity}")

def analyze_training_info(training_info):
    """Analyze training information"""
    
    print("\n📚 TRAINING INFORMATION:")
    
    for key, value in training_info.items():
        if isinstance(value, (int, float)):
            if 'loss' in key.lower():
                print(f"   {key}: {value:.6f}")
            elif 'accuracy' in key.lower() or 'f1' in key.lower():
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")

def analyze_feature_selection(selected_features):
    """Analyze feature selection"""
    
    print(f"\n🎯 FEATURE SELECTION:")
    print(f"   Selected features: {len(selected_features)}")
    
    if hasattr(selected_features, '__iter__') and not isinstance(selected_features, str):
        print("   Features:")
        for i, feature in enumerate(selected_features):
            print(f"     {i+1:2d}. {feature}")

def create_susceptibility_plots(data, output_prefix):
    """Create visualization plots for susceptibility analysis"""
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Susceptibility Map Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        axes[0, 0].axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.3f}')
        axes[0, 0].set_xlabel('Susceptibility Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Susceptibility Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(data, vert=True)
        axes[0, 1].set_ylabel('Susceptibility Value')
        axes[0, 1].set_title('Susceptibility Value Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 0].plot(sorted_data, y, linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Susceptibility Value')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk level pie chart
        very_low = (data <= 0.2).sum()
        low = ((data > 0.2) & (data <= 0.4)).sum()
        moderate = ((data > 0.4) & (data <= 0.6)).sum()
        high = ((data > 0.6) & (data <= 0.8)).sum()
        very_high = (data > 0.8).sum()
        
        sizes = [very_low, low, moderate, high, very_high]
        labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Risk Level Distribution')
        
        plt.tight_layout()
        
        # Save plot
        output_path = f"{output_prefix}_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Susceptibility analysis plots saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def create_performance_plots(landslide_values, full_raster_sample):
    """Create performance comparison plots"""
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Landslide Prediction Performance Analysis', fontsize=16, fontweight='bold')
        
        # Sample full raster for comparison (to avoid memory issues)
        sample_size = min(10000, len(full_raster_sample))
        raster_sample = np.random.choice(full_raster_sample, sample_size)
        
        # Comparison histogram
        axes[0].hist(raster_sample, bins=30, alpha=0.5, label='Full Raster', color='lightblue', density=True)
        axes[0].hist(landslide_values, bins=30, alpha=0.7, label='Landslide Locations', color='red', density=True)
        axes[0].set_xlabel('Susceptibility Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Susceptibility Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        box_data = [raster_sample, landslide_values]
        box_labels = ['Full Raster\n(Sample)', 'Landslide\nLocations']
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('Susceptibility Value')
        axes[1].set_title('Distribution Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # Performance metrics
        risk_levels = ['Very Low\n(≤0.2)', 'Low\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 'High\n(0.6-0.8)', 'Very High\n(>0.8)']
        
        very_low = (landslide_values <= 0.2).sum()
        low = ((landslide_values > 0.2) & (landslide_values <= 0.4)).sum()
        moderate = ((landslide_values > 0.4) & (landslide_values <= 0.6)).sum()
        high = ((landslide_values > 0.6) & (landslide_values <= 0.8)).sum()
        very_high = (landslide_values > 0.8).sum()
        
        counts = [very_low, low, moderate, high, very_high]
        percentages = [c/len(landslide_values)*100 for c in counts]
        
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        bars = axes[2].bar(risk_levels, percentages, color=colors, alpha=0.7, edgecolor='black')
        axes[2].set_ylabel('Percentage of Landslides (%)')
        axes[2].set_title('Landslides by Predicted Risk Level')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = "performance_analysis_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance analysis plots saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create performance plots: {e}")

def generate_recommendations():
    """Generate improvement recommendations based on analysis"""
    
    recommendations = [
        "🎯 MODEL ARCHITECTURE IMPROVEMENTS:",
        "   • Consider ensemble methods (Random Forest + ANN)",
        "   • Implement attention mechanisms for spatial features",
        "   • Add residual connections for deeper networks",
        "   • Experiment with different activation functions (Swish, GELU)",
        "",
        "📊 DATA IMPROVEMENTS:",
        "   • Increase training data diversity (more geographic regions)",
        "   • Add temporal landslide data if available",
        "   • Include additional environmental factors (precipitation, vegetation)",
        "   • Balance positive/negative samples spatially",
        "",
        "🔧 TRAINING IMPROVEMENTS:",
        "   • Implement spatial cross-validation",
        "   • Use focal loss for imbalanced data",
        "   • Add data augmentation (rotation, scaling)",
        "   • Implement early stopping and learning rate scheduling",
        "",
        "🗺️  PREDICTION IMPROVEMENTS:",
        "   • Add uncertainty quantification",
        "   • Implement ensemble predictions",
        "   • Use sliding window predictions for better spatial consistency",
        "   • Add post-processing smoothing filters",
        "",
        "📈 EVALUATION IMPROVEMENTS:",
        "   • Implement ROC/AUC analysis",
        "   • Add spatial autocorrelation analysis",
        "   • Create precision-recall curves",
        "   • Perform cross-validation with different regions"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    # Install missing dependencies
    missing_deps = []
    
    try:
        import rasterio
    except ImportError:
        missing_deps.append("rasterio")
    
    try:
        import geopandas
    except ImportError:
        missing_deps.append("geopandas")
    
    if missing_deps:
        print("Installing missing dependencies...")
        for dep in missing_deps:
            os.system(f"pip install {dep}")
        print("Dependencies installed. Re-run the script.")
        sys.exit(1)
    
    analyze_susceptibility_performance()