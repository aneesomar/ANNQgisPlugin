#!/usr/bin/env python3
"""
Model Diagnostic Analysis - Deep dive into model behavior and feature importance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def detailed_model_analysis():
    """Perform detailed model diagnostic analysis"""
    
    print("=" * 80)
    print("DETAILED MODEL DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    model_path = "/home/anees/Projects/annlandslide_train/outputs/output.pth"
    
    # Load model
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("🔍 DETAILED MODEL ANALYSIS")
    print("-" * 50)
    
    # 1. Analyze layer weights and biases
    analyze_layer_weights(model_data['model_state_dict'])
    
    # 2. Analyze feature importance
    analyze_feature_importance(model_data)
    
    # 3. Analyze training convergence
    analyze_training_convergence(model_data['training_info'])
    
    # 4. Provide specific improvement recommendations
    provide_specific_recommendations(model_data)

def analyze_layer_weights(state_dict):
    """Analyze neural network layer weights for insights"""
    
    print("\n🧠 LAYER-BY-LAYER ANALYSIS:")
    print("-" * 30)
    
    layers = {}
    
    # Group weights by layer
    for name, param in state_dict.items():
        if 'weight' in name:
            layer_num = name.split('.')[1]
            layer_name = f"Layer {layer_num}"
            
            if 'network' in name:
                layers[layer_name] = param.numpy()
    
    # Analyze each layer
    for layer_name, weights in layers.items():
        print(f"\n📊 {layer_name}:")
        print(f"   Shape: {weights.shape}")
        print(f"   Mean weight: {weights.mean():.6f}")
        print(f"   Std weight:  {weights.std():.6f}")
        print(f"   Min weight:  {weights.min():.6f}")
        print(f"   Max weight:  {weights.max():.6f}")
        
        # Check for potential issues
        dead_neurons = (np.abs(weights) < 1e-6).all(axis=1 if len(weights.shape) > 1 else 0).sum()
        if dead_neurons > 0:
            print(f"   ⚠️  Dead neurons: {dead_neurons}")
        
        large_weights = (np.abs(weights) > 10).sum()
        if large_weights > 0:
            print(f"   ⚠️  Large weights (>10): {large_weights}")
    
    # Create weight distribution plots
    create_weight_distribution_plots(layers)

def analyze_feature_importance(model_data):
    """Analyze feature importance and selection"""
    
    print("\n🎯 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 35)
    
    selected_features = model_data['selected_features']
    
    # Get first layer weights (connection to input features)
    first_layer_weights = None
    for name, param in model_data['model_state_dict'].items():
        if 'network.0.weight' in name:
            first_layer_weights = param.numpy()
            break
    
    if first_layer_weights is not None:
        # Calculate feature importance based on first layer weights
        feature_importance = np.abs(first_layer_weights).mean(axis=0)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"📈 TOP 10 MOST IMPORTANT FEATURES:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {i+1:2d}. {row['Feature']:<25} ({row['Importance']:.4f})")
        
        print(f"\n📉 BOTTOM 5 LEAST IMPORTANT FEATURES:")
        for i, row in importance_df.tail(5).iterrows():
            print(f"   {len(importance_df)-4+i:2d}. {row['Feature']:<25} ({row['Importance']:.4f})")
        
        # Analyze feature categories
        analyze_feature_categories(importance_df)
        
        # Create feature importance plot
        create_feature_importance_plot(importance_df)

def analyze_feature_categories(importance_df):
    """Analyze importance by feature categories"""
    
    print(f"\n📊 IMPORTANCE BY FEATURE CATEGORY:")
    
    categories = {
        'Lithology': [f for f in importance_df['Feature'] if 'lithology' in f.lower()],
        'Soil': [f for f in importance_df['Feature'] if 'soil' in f.lower()],
        'Topographic': [f for f in importance_df['Feature'] if any(x in f.lower() for x in ['tri', 'tpi', 'twi', 'spi', 'slope', 'aspect', 'curv', 'dem'])],
        'Hydrologic': [f for f in importance_df['Feature'] if any(x in f.lower() for x in ['flowacc', 'river', 'twi'])],
        'Proximity': [f for f in importance_df['Feature'] if 'distance' in f.lower()]
    }
    
    for category, features in categories.items():
        if features:
            cat_importance = importance_df[importance_df['Feature'].isin(features)]['Importance']
            print(f"   {category:<12}: {len(features):2d} features, avg importance: {cat_importance.mean():.4f}")

def analyze_training_convergence(training_info):
    """Analyze training convergence and stability"""
    
    print(f"\n📈 TRAINING CONVERGENCE ANALYSIS:")
    print("-" * 35)
    
    train_losses = training_info.get('train_losses', [])
    val_losses = training_info.get('val_losses', [])
    
    if train_losses and val_losses:
        print(f"📊 Training Statistics:")
        print(f"   Epochs trained: {len(train_losses)}")
        print(f"   Final train loss: {train_losses[-1]:.6f}")
        print(f"   Final val loss: {val_losses[-1]:.6f}")
        print(f"   Best val loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses))+1})")
        
        # Check for overfitting
        final_gap = val_losses[-1] - train_losses[-1]
        avg_gap = np.mean(np.array(val_losses) - np.array(train_losses))
        
        print(f"\n🎯 Overfitting Analysis:")
        print(f"   Final train-val gap: {final_gap:.6f}")
        print(f"   Average train-val gap: {avg_gap:.6f}")
        
        if final_gap > 0.1:
            print(f"   ⚠️  Possible overfitting detected!")
        elif final_gap < 0.02:
            print(f"   ✅ Good generalization")
        
        # Check convergence
        last_10_val = val_losses[-10:] if len(val_losses) >= 10 else val_losses
        val_stability = np.std(last_10_val)
        
        print(f"\n📊 Convergence Analysis:")
        print(f"   Validation loss stability (last 10 epochs): {val_stability:.6f}")
        
        if val_stability > 0.02:
            print(f"   ⚠️  Training may not have converged properly")
        else:
            print(f"   ✅ Training appears to have converged")
        
        # Create training plots
        create_training_plots(train_losses, val_losses)

def create_weight_distribution_plots(layers):
    """Create weight distribution visualization"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Network Weight Analysis', fontsize=16, fontweight='bold')
        
        layer_names = list(layers.keys())[:4]  # First 4 layers
        
        for i, layer_name in enumerate(layer_names):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            weights = layers[layer_name].flatten()
            
            axes[row, col].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].axvline(weights.mean(), color='red', linestyle='--', 
                                 label=f'Mean: {weights.mean():.3f}')
            axes[row, col].axvline(0, color='green', linestyle='-', alpha=0.5, label='Zero')
            axes[row, col].set_title(f'{layer_name} Weight Distribution')
            axes[row, col].set_xlabel('Weight Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weight_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Weight distribution analysis saved: weight_distribution_analysis.png")
        
    except Exception as e:
        print(f"⚠️  Could not create weight plots: {e}")

def create_feature_importance_plot(importance_df):
    """Create feature importance visualization"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Top features bar plot
        top_features = importance_df.head(15)
        bars = ax1.barh(range(len(top_features)), top_features['Importance'], color='skyblue', edgecolor='black')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'], fontsize=10)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 15 Most Important Features')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Feature importance distribution
        ax2.hist(importance_df['Importance'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(importance_df['Importance'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {importance_df["Importance"].mean():.3f}')
        ax2.axvline(importance_df['Importance'].median(), color='blue', linestyle='--', 
                   label=f'Median: {importance_df["Importance"].median():.3f}')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Importance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance analysis saved: feature_importance_analysis.png")
        
    except Exception as e:
        print(f"⚠️  Could not create feature importance plots: {e}")

def create_training_plots(train_losses, val_losses):
    """Create training convergence visualization"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark best validation loss
        best_epoch = val_losses.index(min(val_losses)) + 1
        ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best Val (Epoch {best_epoch})')
        ax1.legend()
        
        # Training-validation gap
        gap = np.array(val_losses) - np.array(train_losses)
        ax2.plot(epochs, gap, 'purple', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.fill_between(epochs, gap, 0, alpha=0.3, color='purple')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation - Training Loss')
        ax2.set_title('Overfitting Monitor (Val - Train Loss)')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations
        if gap[-1] > 0.1:
            ax2.text(len(epochs)*0.7, max(gap)*0.8, '⚠️ Possible\nOverfitting', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('training_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training convergence analysis saved: training_convergence_analysis.png")
        
    except Exception as e:
        print(f"⚠️  Could not create training plots: {e}")

def provide_specific_recommendations(model_data):
    """Provide specific recommendations based on analysis"""
    
    print(f"\n🎯 SPECIFIC IMPROVEMENT RECOMMENDATIONS:")
    print("-" * 45)
    
    training_info = model_data.get('training_info', {})
    
    # Performance-based recommendations
    f1_score = training_info.get('best_f1', 0)
    auc_score = training_info.get('auc_roc', 0)
    accuracy = training_info.get('accuracy', 0)
    
    print(f"📊 Current Performance:")
    print(f"   F1 Score: {f1_score:.4f}")
    print(f"   AUC-ROC:  {auc_score:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print()
    
    recommendations = []
    
    if f1_score < 0.6:
        recommendations.append("🔧 F1 Score Improvement:")
        recommendations.append("   • Implement class balancing techniques (SMOTE, class weights)")
        recommendations.append("   • Optimize classification threshold")
        recommendations.append("   • Add focal loss for imbalanced data")
    
    if auc_score < 0.85:
        recommendations.append("📈 AUC Score Improvement:")
        recommendations.append("   • Feature engineering: create interaction features")
        recommendations.append("   • Ensemble methods: combine with Random Forest")
        recommendations.append("   • Add more diverse training data")
    
    # Architecture recommendations
    total_params = sum(p.numel() for p in model_data['model_state_dict'].values() if 'weight' in str(p))
    
    if total_params < 30000:
        recommendations.append("🧠 Model Complexity:")
        recommendations.append("   • Increase model capacity (more layers/neurons)")
        recommendations.append("   • Add skip connections for deeper networks")
    elif total_params > 100000:
        recommendations.append("⚡ Model Efficiency:")
        recommendations.append("   • Consider model pruning")
        recommendations.append("   • Implement dropout regularization")
    
    # Training recommendations
    val_losses = training_info.get('val_losses', [])
    if val_losses:
        final_gap = val_losses[-1] - training_info.get('train_losses', [0])[-1]
        if final_gap > 0.1:
            recommendations.append("🎯 Overfitting Mitigation:")
            recommendations.append("   • Increase dropout rate")
            recommendations.append("   • Add L2 regularization")
            recommendations.append("   • Early stopping with patience")
            recommendations.append("   • Data augmentation")
    
    # Feature-based recommendations
    recommendations.append("🎯 Feature Engineering:")
    recommendations.append("   • Create terrain roughness index combinations")
    recommendations.append("   • Add seasonal precipitation data")
    recommendations.append("   • Include vegetation indices (NDVI)")
    recommendations.append("   • Create distance-weighted features")
    
    recommendations.append("📊 Data Quality:")
    recommendations.append("   • Validate categorical encoding consistency")
    recommendations.append("   • Check for spatial autocorrelation in residuals")
    recommendations.append("   • Balance training data spatially")
    recommendations.append("   • Add temporal validation splits")
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    detailed_model_analysis()