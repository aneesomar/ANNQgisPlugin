#!/usr/bin/env python3
"""
ANN Landslide Susceptibility Map Validation Against Historical Landslides
=========================================================================

This script validates the generated susceptibility map against known historical 
landslide locations to assess model performance in real-world conditions.

Author: GitHub Copilot
Date: October 13, 2025
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class LandslideSusceptibilityValidator:
    """
    Validates ANN landslide susceptibility maps against historical landslide data
    """
    
    def __init__(self, susceptibility_map_path, landslide_points_path, threshold=0.405):
        """
        Initialize the validator
        
        Parameters:
        -----------
        susceptibility_map_path : str
            Path to the susceptibility raster map
        landslide_points_path : str  
            Path to historical landslide points (gpkg/shp)
        threshold : float
            Susceptibility threshold for binary classification (default: 0.405)
        """
        self.susceptibility_path = susceptibility_map_path
        self.landslide_points_path = landslide_points_path
        self.threshold = threshold
        self.susceptibility_map = None
        self.landslide_points = None
        self.validation_results = {}
        
    def load_data(self):
        """Load susceptibility map and landslide points"""
        print("📂 Loading validation data...")
        
        # Load susceptibility map
        try:
            with rasterio.open(self.susceptibility_path) as src:
                self.susceptibility_map = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.nodata = src.nodata
                print(f"   ✅ Susceptibility map loaded: {self.susceptibility_map.shape}")
                print(f"   📊 Value range: {np.nanmin(self.susceptibility_map):.3f} - {np.nanmax(self.susceptibility_map):.3f}")
        except Exception as e:
            print(f"   ❌ Error loading susceptibility map: {e}")
            return False
            
        # Load landslide points
        try:
            self.landslide_points = gpd.read_file(self.landslide_points_path)
            print(f"   ✅ Landslide points loaded: {len(self.landslide_points)} points")
            print(f"   📍 CRS: {self.landslide_points.crs}")
            
            # Reproject if needed
            if self.landslide_points.crs != self.crs:
                print(f"   🔄 Reprojecting landslide points to match raster CRS...")
                self.landslide_points = self.landslide_points.to_crs(self.crs)
                
        except Exception as e:
            print(f"   ❌ Error loading landslide points: {e}")
            return False
            
        return True
        
    def extract_susceptibility_at_points(self):
        """Extract susceptibility values at landslide locations"""
        print("\n🎯 Extracting susceptibility values at landslide locations...")
        
        susceptibility_values = []
        valid_points = 0
        
        for idx, point in self.landslide_points.iterrows():
            try:
                # Get pixel coordinates
                col, row = rasterio.transform.rowcol(self.transform, point.geometry.x, point.geometry.y)
                
                # Check if within raster bounds
                if (0 <= row < self.susceptibility_map.shape[0] and 
                    0 <= col < self.susceptibility_map.shape[1]):
                    
                    value = self.susceptibility_map[row, col]
                    
                    # Check for valid value (not nodata)
                    if not np.isnan(value) and value != self.nodata:
                        susceptibility_values.append(value)
                        valid_points += 1
                    else:
                        susceptibility_values.append(np.nan)
                else:
                    susceptibility_values.append(np.nan)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing point {idx}: {e}")
                susceptibility_values.append(np.nan)
        
        self.landslide_susceptibility = np.array(susceptibility_values)
        print(f"   ✅ Extracted values for {valid_points}/{len(self.landslide_points)} points")
        
        return valid_points > 0
        
    def generate_random_non_landslide_points(self, n_points=None):
        """Generate random points in non-landslide areas for comparison"""
        print("\n🎲 Generating random non-landslide points...")
        
        if n_points is None:
            n_points = len(self.landslide_points) * 3  # 3x more non-landslide points
            
        # Get valid (non-nodata) pixel locations
        valid_mask = ~np.isnan(self.susceptibility_map) & (self.susceptibility_map != self.nodata)
        valid_rows, valid_cols = np.where(valid_mask)
        
        # Create buffer around landslide points to avoid sampling too close
        buffer_distance = 100  # meters
        buffer_pixels = int(buffer_distance / abs(self.transform[0]))  # approximate pixel buffer
        
        # Create exclusion mask around landslide points
        exclusion_mask = np.zeros_like(self.susceptibility_map, dtype=bool)
        
        for idx, point in self.landslide_points.iterrows():
            try:
                col, row = rasterio.transform.rowcol(self.transform, point.geometry.x, point.geometry.y)
                if (0 <= row < self.susceptibility_map.shape[0] and 
                    0 <= col < self.susceptibility_map.shape[1]):
                    
                    # Create circular exclusion zone
                    r_min = max(0, row - buffer_pixels)
                    r_max = min(self.susceptibility_map.shape[0], row + buffer_pixels + 1)
                    c_min = max(0, col - buffer_pixels)
                    c_max = min(self.susceptibility_map.shape[1], col + buffer_pixels + 1)
                    
                    exclusion_mask[r_min:r_max, c_min:c_max] = True
            except:
                continue
        
        # Get valid sampling locations (valid pixels not near landslides)
        sampling_mask = valid_mask & ~exclusion_mask
        sampling_rows, sampling_cols = np.where(sampling_mask)
        
        if len(sampling_rows) < n_points:
            n_points = len(sampling_rows)
            print(f"   ⚠️ Reduced random points to {n_points} (limited by valid area)")
        
        # Randomly sample points
        random_indices = np.random.choice(len(sampling_rows), size=n_points, replace=False)
        random_rows = sampling_rows[random_indices]
        random_cols = sampling_cols[random_indices]
        
        # Extract susceptibility values
        self.non_landslide_susceptibility = self.susceptibility_map[random_rows, random_cols]
        
        print(f"   ✅ Generated {len(self.non_landslide_susceptibility)} random non-landslide points")
        print(f"   📊 Non-landslide susceptibility range: {np.min(self.non_landslide_susceptibility):.3f} - {np.max(self.non_landslide_susceptibility):.3f}")
        
        return True
        
    def calculate_validation_metrics(self):
        """Calculate comprehensive validation metrics"""
        print("\n📊 Calculating validation metrics...")
        
        # Remove NaN values from landslide points
        valid_landslide_mask = ~np.isnan(self.landslide_susceptibility)
        valid_landslide_values = self.landslide_susceptibility[valid_landslide_mask]
        
        if len(valid_landslide_values) == 0:
            print("   ❌ No valid landslide points found!")
            return False
        
        # Combine landslide and non-landslide data
        y_true = np.concatenate([
            np.ones(len(valid_landslide_values)),      # 1 = landslide
            np.zeros(len(self.non_landslide_susceptibility))  # 0 = non-landslide
        ])
        
        y_proba = np.concatenate([
            valid_landslide_values,
            self.non_landslide_susceptibility
        ])
        
        # Binary predictions using threshold
        y_pred = (y_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        self.validation_results = {
            'n_landslide_points': len(valid_landslide_values),
            'n_non_landslide_points': len(self.non_landslide_susceptibility),
            'threshold': self.threshold,
            
            # Landslide-specific statistics
            'landslide_susceptibility_mean': np.mean(valid_landslide_values),
            'landslide_susceptibility_median': np.median(valid_landslide_values),
            'landslide_susceptibility_std': np.std(valid_landslide_values),
            'landslides_above_threshold': np.sum(valid_landslide_values >= self.threshold),
            'landslides_above_threshold_pct': np.sum(valid_landslide_values >= self.threshold) / len(valid_landslide_values) * 100,
            
            # Non-landslide statistics  
            'non_landslide_susceptibility_mean': np.mean(self.non_landslide_susceptibility),
            'non_landslide_susceptibility_median': np.median(self.non_landslide_susceptibility),
            'non_landslide_susceptibility_std': np.std(self.non_landslide_susceptibility),
            
            # Classification metrics
            'accuracy': np.mean(y_pred == y_true),
            'true_positives': np.sum((y_pred == 1) & (y_true == 1)),
            'false_positives': np.sum((y_pred == 1) & (y_true == 0)),
            'true_negatives': np.sum((y_pred == 0) & (y_true == 0)),
            'false_negatives': np.sum((y_pred == 0) & (y_true == 1)),
        }
        
        # Calculate derived metrics
        tp = self.validation_results['true_positives']
        fp = self.validation_results['false_positives']
        tn = self.validation_results['true_negatives']
        fn = self.validation_results['false_negatives']
        
        self.validation_results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.validation_results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.validation_results['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        self.validation_results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC and PR curves
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            self.validation_results['auc_roc'] = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            self.validation_results['auc_pr'] = average_precision_score(y_true, y_proba)
            
        except Exception as e:
            print(f"   ⚠️ Error calculating AUC metrics: {e}")
            self.validation_results['auc_roc'] = 0
            self.validation_results['auc_pr'] = 0
        
        print("   ✅ Validation metrics calculated successfully")
        return True
        
    def print_validation_report(self):
        """Print comprehensive validation report"""
        print("\n" + "="*80)
        print("🏆 LANDSLIDE SUSCEPTIBILITY MAP VALIDATION REPORT")
        print("="*80)
        
        results = self.validation_results
        
        print(f"\n📍 DATA SUMMARY:")
        print(f"   Historical landslide points: {results['n_landslide_points']:,}")
        print(f"   Random non-landslide points: {results['n_non_landslide_points']:,}")
        print(f"   Classification threshold: {results['threshold']:.3f}")
        
        print(f"\n📊 SUSCEPTIBILITY AT LANDSLIDE LOCATIONS:")
        print(f"   Mean susceptibility: {results['landslide_susceptibility_mean']:.3f}")
        print(f"   Median susceptibility: {results['landslide_susceptibility_median']:.3f}")
        print(f"   Standard deviation: {results['landslide_susceptibility_std']:.3f}")
        print(f"   Above threshold: {results['landslides_above_threshold']}/{results['n_landslide_points']} ({results['landslides_above_threshold_pct']:.1f}%)")
        
        print(f"\n📊 SUSCEPTIBILITY AT NON-LANDSLIDE LOCATIONS:")
        print(f"   Mean susceptibility: {results['non_landslide_susceptibility_mean']:.3f}")
        print(f"   Median susceptibility: {results['non_landslide_susceptibility_median']:.3f}")
        print(f"   Standard deviation: {results['non_landslide_susceptibility_std']:.3f}")
        
        print(f"\n🎯 VALIDATION METRICS:")
        print(f"   Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"   Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
        print(f"   Recall (Sensitivity): {results['recall']:.3f} ({results['recall']*100:.1f}%)")
        print(f"   Specificity: {results['specificity']:.3f} ({results['specificity']*100:.1f}%)")
        print(f"   F1 Score: {results['f1_score']:.3f}")
        print(f"   AUC-ROC: {results['auc_roc']:.3f}")
        print(f"   PR-AUC: {results['auc_pr']:.3f}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"                    Predicted")
        print(f"                 Low    High")
        print(f"   Actual Low   {results['true_negatives']:4}   {results['false_positives']:4}")
        print(f"   Actual High  {results['false_negatives']:4}   {results['true_positives']:4}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if results['landslides_above_threshold_pct'] >= 80:
            print("   ✅ EXCELLENT: >80% of historical landslides in high susceptibility areas")
        elif results['landslides_above_threshold_pct'] >= 70:
            print("   ✅ VERY GOOD: 70-80% of historical landslides captured")
        elif results['landslides_above_threshold_pct'] >= 60:
            print("   ⚠️ GOOD: 60-70% of historical landslides captured")
        elif results['landslides_above_threshold_pct'] >= 50:
            print("   ⚠️ FAIR: 50-60% of historical landslides captured")
        else:
            print("   ❌ POOR: <50% of historical landslides captured - model needs improvement")
            
        if results['auc_roc'] >= 0.8:
            print("   ✅ EXCELLENT discriminative ability (AUC-ROC ≥ 0.8)")
        elif results['auc_roc'] >= 0.7:
            print("   ✅ GOOD discriminative ability (AUC-ROC ≥ 0.7)")
        elif results['auc_roc'] >= 0.6:
            print("   ⚠️ FAIR discriminative ability (AUC-ROC ≥ 0.6)")
        else:
            print("   ❌ POOR discriminative ability (AUC-ROC < 0.6)")
            
        print("="*80)
        
    def create_validation_plots(self, output_dir="validation_plots"):
        """Create visualization plots for validation results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📈 Creating validation plots in {output_dir}/...")
        
        # Remove NaN values
        valid_landslide_mask = ~np.isnan(self.landslide_susceptibility)
        valid_landslide_values = self.landslide_susceptibility[valid_landslide_mask]
        
        # 1. Susceptibility distribution comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(valid_landslide_values, bins=30, alpha=0.7, label='Landslide locations', color='red', density=True)
        plt.hist(self.non_landslide_susceptibility, bins=30, alpha=0.7, label='Non-landslide locations', color='blue', density=True)
        plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.3f})')
        plt.xlabel('Susceptibility Value')
        plt.ylabel('Density')
        plt.title('Susceptibility Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        plt.subplot(2, 2, 2)
        data_to_plot = [valid_landslide_values, self.non_landslide_susceptibility]
        labels = ['Landslide\nLocations', 'Non-landslide\nLocations']
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][1].set_facecolor('blue')
        plt.axhline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.3f})')
        plt.ylabel('Susceptibility Value')
        plt.title('Susceptibility Value Distributions')
        plt.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_landslide = np.sort(valid_landslide_values)
        sorted_non_landslide = np.sort(self.non_landslide_susceptibility)
        
        plt.plot(sorted_landslide, np.arange(1, len(sorted_landslide) + 1) / len(sorted_landslide), 
                'r-', linewidth=2, label='Landslide locations')
        plt.plot(sorted_non_landslide, np.arange(1, len(sorted_non_landslide) + 1) / len(sorted_non_landslide), 
                'b-', linewidth=2, label='Non-landslide locations')
        plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.3f})')
        plt.xlabel('Susceptibility Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Performance metrics bar plot
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'PR-AUC']
        values = [
            self.validation_results['accuracy'],
            self.validation_results['precision'], 
            self.validation_results['recall'],
            self.validation_results['f1_score'],
            self.validation_results['auc_roc'],
            self.validation_results['auc_pr']
        ]
        
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in values]
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, 1)
        plt.title('Validation Performance Metrics')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/landslide_validation_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Validation plots saved to {output_dir}/")
        
    def run_full_validation(self):
        """Run complete validation workflow"""
        print("🚀 Starting landslide susceptibility map validation...")
        
        # Load data
        if not self.load_data():
            return False
            
        # Extract susceptibility values at landslide locations
        if not self.extract_susceptibility_at_points():
            return False
            
        # Generate random non-landslide points
        if not self.generate_random_non_landslide_points():
            return False
            
        # Calculate validation metrics
        if not self.calculate_validation_metrics():
            return False
            
        # Print results
        self.print_validation_report()
        
        # Create plots
        try:
            self.create_validation_plots()
        except Exception as e:
            print(f"   ⚠️ Could not create plots: {e}")
        
        print("\n✅ Validation completed successfully!")
        return True


def main():
    """Main validation function"""
    print("🗺️ ANN Landslide Susceptibility Map Validation")
    print("=" * 60)
    
    # File paths
    susceptibility_map = "/home/anees/Projects/annlandslide_train/outputs/map5"
    landslide_points = "/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/DurbanRasters/clipped_landslidePoints_lo19.gpkg"
    threshold = 0.405  # From your training results
    
    # Initialize validator
    validator = LandslideSusceptibilityValidator(
        susceptibility_map_path=susceptibility_map,
        landslide_points_path=landslide_points,
        threshold=threshold
    )
    
    # Run validation
    success = validator.run_full_validation()
    
    if success:
        print(f"\n🎯 Key Result: {validator.validation_results['landslides_above_threshold_pct']:.1f}% of historical landslides correctly identified as high risk!")
    
    return validator

if __name__ == "__main__":
    validator = main()