#!/usr/bin/env python3
"""
Data Quality Analysis for Landslide Susceptibility Modeling
===========================================================

This script analyzes the quality of training data and identifies potential
issues that may be limiting model performance.

Author: GitHub Copilot
Date: October 13, 2025
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Comprehensive analysis of training data quality issues"""
    
    print("🔍 DATA QUALITY ANALYSIS FOR LANDSLIDE MODELING")
    print("=" * 60)
    
    # Load training data
    print("📂 Loading training data...")
    try:
        X_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_train.csv')
        y_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_train.csv')
        X_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_test.csv')
        y_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_test.csv')
        
        print(f"   ✅ Training samples: {len(X_train):,}")
        print(f"   ✅ Test samples: {len(X_test):,}")
        print(f"   ✅ Features: {X_train.shape[1]}")
        
    except Exception as e:
        print(f"   ❌ Error loading training data: {e}")
        return False
    
    # Combine for analysis
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True).iloc[:, 0]  # First column
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total samples: {len(X_all):,}")
    print(f"   Landslide samples: {np.sum(y_all == 1):,} ({np.mean(y_all == 1)*100:.1f}%)")
    print(f"   Non-landslide samples: {np.sum(y_all == 0):,} ({np.mean(y_all == 0)*100:.1f}%)")
    
    # Feature names (assuming standard geomorphological features)
    feature_names = [
        'Elevation', 'Slope', 'Aspect', 'Plan_Curvature', 'Profile_Curvature',
        'Flow_Accumulation', 'TWI', 'SPI', 'TPI', 'TRI', 
        'Distance_to_Rivers', 'Distance_to_Roads', 'Lithology', 'Soil_Type'
    ]
    
    # Use actual column names if available
    if len(X_all.columns) <= len(feature_names):
        X_all.columns = feature_names[:len(X_all.columns)]
    
    # 1. CLASS IMBALANCE ANALYSIS
    print(f"\n⚖️ CLASS IMBALANCE ANALYSIS:")
    landslide_ratio = np.mean(y_all == 1)
    
    if landslide_ratio < 0.01:
        print(f"   ❌ SEVERE IMBALANCE: {landslide_ratio*100:.1f}% landslides - model will struggle")
        imbalance_severity = "SEVERE"
    elif landslide_ratio < 0.05:
        print(f"   ⚠️ HIGH IMBALANCE: {landslide_ratio*100:.1f}% landslides - challenging but manageable")
        imbalance_severity = "HIGH"
    elif landslide_ratio < 0.1:
        print(f"   ⚠️ MODERATE IMBALANCE: {landslide_ratio*100:.1f}% landslides")
        imbalance_severity = "MODERATE"
    else:
        print(f"   ✅ REASONABLE BALANCE: {landslide_ratio*100:.1f}% landslides")
        imbalance_severity = "ACCEPTABLE"
    
    # 2. FEATURE QUALITY ANALYSIS
    print(f"\n📊 FEATURE QUALITY ANALYSIS:")
    
    quality_issues = []
    
    for i, feature in enumerate(X_all.columns):
        col_data = X_all.iloc[:, i]
        
        # Check for missing values
        missing_pct = col_data.isnull().sum() / len(col_data) * 100
        
        # Check for constant values
        unique_vals = col_data.nunique()
        constant_risk = unique_vals < 10
        
        # Check for extreme outliers (more than 3 IQR from median)
        Q1, Q3 = col_data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_threshold = 3 * IQR
        outliers = ((col_data < (Q1 - outlier_threshold)) | (col_data > (Q3 + outlier_threshold))).sum()
        outlier_pct = outliers / len(col_data) * 100
        
        # Check variance
        variance = col_data.var()
        low_variance = variance < 0.01
        
        print(f"   {feature}:")
        print(f"      Missing: {missing_pct:.1f}%, Unique: {unique_vals:,}, Variance: {variance:.3f}")
        
        if missing_pct > 5:
            quality_issues.append(f"{feature}: {missing_pct:.1f}% missing values")
            print(f"      ❌ High missing values")
        
        if constant_risk:
            quality_issues.append(f"{feature}: Only {unique_vals} unique values")
            print(f"      ⚠️ Low diversity")
            
        if outlier_pct > 10:
            quality_issues.append(f"{feature}: {outlier_pct:.1f}% extreme outliers")
            print(f"      ⚠️ Many outliers")
            
        if low_variance:
            quality_issues.append(f"{feature}: Very low variance ({variance:.3f})")
            print(f"      ⚠️ Low variance")
    
    # 3. SEPARABILITY ANALYSIS
    print(f"\n🎯 CLASS SEPARABILITY ANALYSIS:")
    
    separability_scores = []
    
    for i, feature in enumerate(X_all.columns):
        landslide_mask = y_all == 1
        non_landslide_mask = y_all == 0
        
        landslide_vals = X_all.loc[landslide_mask, feature].dropna()
        non_landslide_vals = X_all.loc[non_landslide_mask, feature].dropna()
        
        if len(landslide_vals) > 10 and len(non_landslide_vals) > 10:
            # Statistical test for difference
            try:
                t_stat, p_val = stats.ttest_ind(landslide_vals, non_landslide_vals)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(landslide_vals) - 1) * landslide_vals.var() + 
                                    (len(non_landslide_vals) - 1) * non_landslide_vals.var()) / 
                                   (len(landslide_vals) + len(non_landslide_vals) - 2))
                cohens_d = abs(landslide_vals.mean() - non_landslide_vals.mean()) / pooled_std
                
                separability_scores.append((feature, p_val, cohens_d))
                
                print(f"   {feature}: p={p_val:.2e}, effect_size={cohens_d:.3f}")
                
            except:
                separability_scores.append((feature, 1.0, 0.0))
                print(f"   {feature}: Unable to calculate")
    
    # Rank features by separability
    separability_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by effect size
    
    print(f"\n🏆 TOP DISCRIMINATIVE FEATURES:")
    for i, (feature, p_val, effect_size) in enumerate(separability_scores[:5]):
        if effect_size > 0.5:
            status = "✅ Strong"
        elif effect_size > 0.2:
            status = "⚠️ Moderate"
        else:
            status = "❌ Weak"
        print(f"   {i+1}. {feature}: {status} (effect size: {effect_size:.3f})")
    
    # 4. SPATIAL AUTOCORRELATION CONCERNS
    print(f"\n🗺️ POTENTIAL SPATIAL ISSUES:")
    
    # Estimate clustering using sample coordinates if available
    try:
        # Check if we have spatial clustering in the data
        sample_indices = np.random.choice(len(X_all), min(1000, len(X_all)), replace=False)
        sample_data = X_all.iloc[sample_indices].values
        
        if len(sample_data) > 100:
            # Use KMeans to detect clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sample_data)
            
            # Check if landslides cluster together
            sample_y = y_all.iloc[sample_indices]
            cluster_landslide_rates = []
            
            for cluster_id in range(5):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 10:
                    cluster_landslide_rate = np.mean(sample_y.iloc[cluster_mask] == 1)
                    cluster_landslide_rates.append(cluster_landslide_rate)
            
            if len(cluster_landslide_rates) > 1:
                cluster_variation = np.std(cluster_landslide_rates)
                if cluster_variation > 0.1:
                    print(f"   ⚠️ HIGH SPATIAL CLUSTERING detected (std: {cluster_variation:.3f})")
                    print(f"      This may lead to overfitting and poor generalization")
                else:
                    print(f"   ✅ Reasonable spatial distribution")
    except:
        print(f"   ⚠️ Unable to assess spatial clustering")
    
    # 5. DATA QUALITY SUMMARY
    print(f"\n📋 DATA QUALITY SUMMARY:")
    print(f"   Class imbalance severity: {imbalance_severity}")
    print(f"   Number of quality issues: {len(quality_issues)}")
    print(f"   Strong discriminative features: {sum(1 for _, _, d in separability_scores if d > 0.5)}")
    
    # 6. RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
    
    if imbalance_severity in ["SEVERE", "HIGH"]:
        print(f"   1. 🎯 ADDRESS CLASS IMBALANCE:")
        print(f"      • Use SMOTE or ADASYN for synthetic oversampling")
        print(f"      • Apply class weights (current ratio: 1:{1/landslide_ratio:.0f})")
        print(f"      • Consider focal loss instead of cross-entropy")
    
    if len(quality_issues) > 3:
        print(f"   2. 🔧 IMPROVE DATA QUALITY:")
        for issue in quality_issues[:3]:
            print(f"      • Fix: {issue}")
        if len(quality_issues) > 3:
            print(f"      • ... and {len(quality_issues) - 3} more issues")
    
    poor_features = [f for f, _, d in separability_scores if d < 0.1]
    if len(poor_features) > 2:
        print(f"   3. 📊 FEATURE ENGINEERING:")
        print(f"      • Consider removing weak features: {', '.join(poor_features[:3])}")
        print(f"      • Add derived features (slope aspect categories, curvature combinations)")
        print(f"      • Apply feature scaling/normalization")
    
    print(f"   4. 🏗️ MODEL ARCHITECTURE:")
    print(f"      • Use ensemble methods (Random Forest, XGBoost)")
    print(f"      • Apply dropout and regularization more aggressively")
    print(f"      • Consider simpler models with current data quality")
    
    print(f"   5. 📍 DATA COLLECTION:")
    print(f"      • Collect more diverse landslide samples")
    print(f"      • Ensure spatial representativeness")
    print(f"      • Validate with independent field surveys")
    
    return {
        'imbalance_severity': imbalance_severity,
        'quality_issues': quality_issues,
        'separability_scores': separability_scores,
        'landslide_ratio': landslide_ratio
    }

def create_data_quality_plots(results):
    """Create visualizations for data quality analysis"""
    
    # Load data again for plotting
    X_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_train.csv')
    y_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_train.csv')
    X_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_test.csv')
    y_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_test.csv')
    
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True).iloc[:, 0]
    
    plt.figure(figsize=(16, 12))
    
    # 1. Class imbalance visualization
    plt.subplot(2, 3, 1)
    class_counts = y_all.value_counts()
    colors = ['lightblue', 'red']
    bars = plt.bar(['Non-landslide', 'Landslide'], class_counts.values, color=colors, alpha=0.7)
    
    for bar, count in zip(bars, class_counts.values):
        pct = count / len(y_all) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(y_all)*0.01, 
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.title('Class Distribution\n(Imbalance Issue)')
    plt.ylabel('Number of Samples')
    
    # 2. Feature correlation heatmap
    plt.subplot(2, 3, 2)
    correlation_matrix = X_all.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlations\n(Check for Redundancy)')
    
    # 3. Feature separability
    plt.subplot(2, 3, 3)
    separability_scores = results['separability_scores']
    
    features = [s[0] for s in separability_scores[:8]]  # Top 8 features
    effect_sizes = [s[2] for s in separability_scores[:8]]
    
    colors = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
    bars = plt.barh(features, effect_sizes, color=colors, alpha=0.7)
    
    plt.axvline(0.2, color='orange', linestyle='--', alpha=0.7, label='Weak')
    plt.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='Strong')
    
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.title('Feature Discriminative Power\n(Higher = Better)')
    plt.legend()
    
    # 4. Distribution comparison for top feature
    plt.subplot(2, 3, 4)
    if len(separability_scores) > 0:
        top_feature = separability_scores[0][0]
        feature_idx = list(X_all.columns).index(top_feature)
        
        landslide_vals = X_all.loc[y_all == 1, top_feature].dropna()
        non_landslide_vals = X_all.loc[y_all == 0, top_feature].dropna()
        
        plt.hist(non_landslide_vals, bins=30, alpha=0.6, label='Non-landslide', 
                color='lightblue', density=True)
        plt.hist(landslide_vals, bins=30, alpha=0.8, label='Landslide', 
                color='red', density=True)
        
        plt.xlabel(f'{top_feature} Value')
        plt.ylabel('Density')
        plt.title(f'Best Separating Feature:\n{top_feature}')
        plt.legend()
    
    # 5. Missing values analysis
    plt.subplot(2, 3, 5)
    missing_pct = X_all.isnull().sum() / len(X_all) * 100
    
    if missing_pct.sum() > 0:
        missing_features = missing_pct[missing_pct > 0]
        plt.bar(range(len(missing_features)), missing_features.values, color='red', alpha=0.7)
        plt.xticks(range(len(missing_features)), missing_features.index, rotation=45)
        plt.ylabel('Missing %')
        plt.title('Missing Values by Feature')
    else:
        plt.text(0.5, 0.5, 'No Missing Values\n✅ Good Data Quality', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
        plt.title('Missing Values Analysis')
    
    # 6. Variance analysis
    plt.subplot(2, 3, 6)
    feature_vars = X_all.var()
    
    plt.bar(range(len(feature_vars)), feature_vars.values, alpha=0.7)
    plt.xticks(range(len(feature_vars)), X_all.columns, rotation=45)
    plt.ylabel('Variance')
    plt.title('Feature Variance\n(Low variance = Less informative)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('validation_plots/data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = analyze_data_quality()
    if results:
        create_data_quality_plots(results)
        print(f"\n✅ Data quality analysis completed!")
        print(f"🎯 Key insight: {results['imbalance_severity']} class imbalance with {results['landslide_ratio']*100:.1f}% landslides")