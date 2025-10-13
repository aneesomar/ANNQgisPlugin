#!/usr/bin/env python3
"""
Comprehensive Model Validation
Tests model against your actual validation/test data with ML metrics
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from pathlib import Path


class SimplifiedLandslideANN(nn.Module):
    """Match the simplified architecture"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.5):
        super(SimplifiedLandslideANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], 1)
        )
    
    def forward(self, x):
        return self.network(x)


def load_test_data(test_data_dir):
    """Load X_test and y_test from CSV files"""
    x_test_path = Path(test_data_dir) / 'X_test_spatial.csv'
    y_test_path = Path(test_data_dir) / 'y_test_spatial.csv'
    
    if not x_test_path.exists():
        print(f"❌ X_test not found at: {x_test_path}")
        print(f"Looking for any test files...")
        test_dir = Path(test_data_dir)
        test_files = list(test_dir.glob('*test*.csv'))
        if test_files:
            print(f"Found these test files:")
            for f in test_files:
                print(f"  - {f.name}")
        return None, None
    
    print(f"📂 Loading test data from: {test_data_dir}")
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    
    # Get labels (last column)
    if 'label' in y_test.columns:
        y_test = y_test['label'].values
    else:
        y_test = y_test.iloc[:, -1].values
    
    print(f"✓ Loaded {len(X_test)} test samples")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  Positive class: {y_test.sum()} ({100*y_test.mean():.1f}%)")
    print(f"  Negative class: {len(y_test) - y_test.sum()} ({100*(1-y_test.mean()):.1f}%)")
    
    return X_test, y_test


def validate_model(model_path, test_data_dir, output_dir=None):
    """
    Comprehensive model validation with metrics
    
    Args:
        model_path: Path to .pth model file
        test_data_dir: Directory containing X_test_spatial.csv and y_test_spatial.csv
        output_dir: Where to save plots (optional)
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_dir}")
    
    # Load test data
    X_test, y_test = load_test_data(test_data_dir)
    if X_test is None:
        return None
    
    # Load model
    print(f"\n📦 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract components
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    scaler = checkpoint.get('scaler')
    selected_features = checkpoint.get('selected_features')
    threshold = checkpoint.get('best_threshold', 0.5)
    
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Scaler: {type(scaler).__name__ if scaler else 'None'}")
    print(f"  Selected features: {len(selected_features) if selected_features else 'All'}")
    
    # Determine model architecture
    if 'network.0.weight' in state_dict:
        input_size = state_dict['network.0.weight'].shape[1]
        print(f"  Architecture: Simplified ({input_size} inputs)")
    elif 'input_layer.0.weight' in state_dict:
        input_size = state_dict['input_layer.0.weight'].shape[1]
        print(f"  Architecture: Complex ({input_size} inputs)")
    else:
        print(f"❌ Cannot determine model architecture")
        return None
    
    # Create model
    model = SimplifiedLandslideANN(input_size=input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded")
    
    # Prepare features
    print(f"\n🔧 Preparing features...")
    
    # Apply feature selection if needed
    if selected_features and len(selected_features) < X_test.shape[1]:
        print(f"  Selecting {len(selected_features)} features from {X_test.shape[1]}")
        feature_names = X_test.columns.tolist()
        
        # Try to match features (handle _aligned suffix differences)
        feature_indices = []
        for selected_feat in selected_features:
            # Try exact match first
            if selected_feat in feature_names:
                feature_indices.append(feature_names.index(selected_feat))
            else:
                # Try without _aligned suffix
                alt_name = selected_feat.replace('_aligned', '')
                if alt_name in feature_names:
                    feature_indices.append(feature_names.index(alt_name))
                # Try with _aligned suffix
                elif f"{selected_feat}_aligned" in feature_names:
                    feature_indices.append(feature_names.index(f"{selected_feat}_aligned"))
                else:
                    print(f"    ⚠️  Feature not found: {selected_feat}")
        
        if len(feature_indices) == len(selected_features):
            X_test = X_test.iloc[:, feature_indices]
            print(f"  ✓ Matched all {len(feature_indices)} features")
        else:
            print(f"  ⚠️ Warning: Only matched {len(feature_indices)}/{len(selected_features)} features")
            print(f"  Using all {X_test.shape[1]} features instead")
    
    # Apply scaling
    if scaler:
        print(f"  Applying {type(scaler).__name__}...")
        # Convert to numpy array to avoid feature name issues
        X_test_scaled = scaler.transform(X_test.values)
    else:
        print(f"  ⚠️ No scaler found, using raw features")
        X_test_scaled = X_test.values
    
    # Make predictions
    print(f"\n🔮 Making predictions on {len(X_test)} samples...")
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
    
    print(f"✓ Predictions complete")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"\n📊 Classification Metrics (at threshold {threshold:.3f}):")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} (of predicted landslides, {precision*100:.1f}% are correct)")
    print(f"  Recall:    {recall:.4f} (finds {recall*100:.1f}% of actual landslides)")
    print(f"  F1 Score:  {f1:.4f}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, probabilities)
        print(f"  ROC-AUC:   {roc_auc:.4f} (1.0 = perfect, 0.5 = random)")
    except:
        roc_auc = None
        print(f"  ROC-AUC:   N/A")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 Confusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"  Actual Negative:    {tn:6d}              {fp:6d}")
    print(f"  Actual Positive:    {fn:6d}              {tp:6d}")
    
    print(f"\n  True Negatives:  {tn:,} (correctly identified non-landslides)")
    print(f"  True Positives:  {tp:,} (correctly identified landslides)")
    print(f"  False Positives: {fp:,} (false alarms)")
    print(f"  False Negatives: {fn:,} (missed landslides) ⚠️")
    
    # Specificity and other metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n  Specificity: {specificity:.4f} (correctly identifies {specificity*100:.1f}% of non-landslides)")
    
    # Prediction distribution
    print(f"\n📈 Prediction Distribution:")
    print(f"  Mean probability: {probabilities.mean():.4f}")
    print(f"  Std deviation:    {probabilities.std():.4f}")
    print(f"  Min probability:  {probabilities.min():.4f}")
    print(f"  Max probability:  {probabilities.max():.4f}")
    print(f"  Median:           {np.median(probabilities):.4f}")
    
    # Distribution by class
    landslide_probs = probabilities[y_test == 1]
    non_landslide_probs = probabilities[y_test == 0]
    
    print(f"\n  Landslide samples (actual=1):")
    print(f"    Mean: {landslide_probs.mean():.4f}, Std: {landslide_probs.std():.4f}")
    print(f"  Non-landslide samples (actual=0):")
    print(f"    Mean: {non_landslide_probs.mean():.4f}, Std: {non_landslide_probs.std():.4f}")
    
    separation = abs(landslide_probs.mean() - non_landslide_probs.mean())
    print(f"  Class separation: {separation:.4f} (higher is better)")
    
    # Threshold analysis
    print(f"\n🎯 Threshold Analysis:")
    print(f"  Current threshold: {threshold:.3f}")
    print(f"    Predicted positive: {predictions.sum():,} ({100*predictions.mean():.1f}%)")
    print(f"    Predicted negative: {len(predictions)-predictions.sum():,} ({100*(1-predictions.mean()):.1f}%)")
    
    # Try different thresholds
    print(f"\n  Alternative thresholds:")
    for alt_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if abs(alt_thresh - threshold) > 0.05:
            alt_preds = (probabilities >= alt_thresh).astype(int)
            alt_f1 = f1_score(y_test, alt_preds)
            alt_recall = recall_score(y_test, alt_preds)
            alt_precision = precision_score(y_test, alt_preds)
            print(f"    {alt_thresh:.1f}: F1={alt_f1:.3f}, Recall={alt_recall:.3f}, Precision={alt_precision:.3f}")
    
    # Assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    issues = []
    good = []
    
    if accuracy >= 0.85:
        good.append(f"✅ Excellent accuracy ({accuracy*100:.1f}%)")
    elif accuracy >= 0.75:
        good.append(f"✅ Good accuracy ({accuracy*100:.1f}%)")
    else:
        issues.append(f"⚠️  Low accuracy ({accuracy*100:.1f}%)")
    
    if roc_auc and roc_auc >= 0.90:
        good.append(f"✅ Excellent ROC-AUC ({roc_auc:.3f})")
    elif roc_auc and roc_auc >= 0.80:
        good.append(f"✅ Good ROC-AUC ({roc_auc:.3f})")
    elif roc_auc:
        issues.append(f"⚠️  Moderate ROC-AUC ({roc_auc:.3f})")
    
    if recall >= 0.80:
        good.append(f"✅ Good recall - catches {recall*100:.1f}% of landslides")
    elif recall >= 0.70:
        issues.append(f"⚠️  Moderate recall - misses {(1-recall)*100:.1f}% of landslides")
    else:
        issues.append(f"❌ Low recall - misses {(1-recall)*100:.1f}% of landslides!")
    
    if precision >= 0.80:
        good.append(f"✅ Good precision - few false alarms")
    elif precision >= 0.70:
        issues.append(f"⚠️  Moderate precision - some false alarms")
    else:
        issues.append(f"⚠️  Low precision - many false alarms")
    
    if probabilities.std() >= 0.15:
        good.append(f"✅ Good prediction variance (Std={probabilities.std():.3f})")
    else:
        issues.append(f"⚠️  Low prediction variance (Std={probabilities.std():.3f}) - too confident")
    
    if separation >= 0.3:
        good.append(f"✅ Good class separation ({separation:.3f})")
    else:
        issues.append(f"⚠️  Classes not well separated ({separation:.3f})")
    
    for msg in good:
        print(msg)
    
    if issues:
        print()
        for msg in issues:
            print(msg)
    
    if not issues:
        print("\n🎉 Model validation PASSED! Model is performing well.")
    elif len(issues) <= 2:
        print("\n✅ Model validation ACCEPTABLE with minor issues.")
    else:
        print("\n⚠️  Model has several issues - consider retraining.")
    
    print("="*70)
    
    # Save plots if output directory specified
    if output_dir:
        save_validation_plots(y_test, probabilities, predictions, threshold, output_dir, model_path)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'threshold': threshold,
        'mean_prob': probabilities.mean(),
        'std_prob': probabilities.std()
    }


def save_validation_plots(y_test, probabilities, predictions, threshold, output_dir, model_path):
    """Save validation plots"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    
    print(f"\n📊 Saving validation plots to: {output_dir}")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{model_name}_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved ROC curve")
    
    # 2. Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities[y_test == 0], bins=50, alpha=0.5, label='Non-landslides', color='green')
    plt.hist(probabilities[y_test == 1], bins=50, alpha=0.5, label='Landslides', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(f'Prediction Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{model_name}_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved distribution plot")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Non-landslide', 'Landslide'])
    plt.yticks(tick_marks, ['Non-landslide', 'Landslide'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]:,}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrix")
    
    print(f"\n✅ All plots saved to: {output_dir}")


def main():
    """Main entry point"""
    
    # Default paths
    MODEL_PATH = "/home/anees/OneDrive/geoProject/Durban/output5.pth"
    TEST_DATA_DIR = "/home/anees/OneDrive/geoProject/python"
    OUTPUT_DIR = "/home/anees/OneDrive/geoProject/Durban/validation_results"
    
    # Parse command line
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        TEST_DATA_DIR = sys.argv[2]
    if len(sys.argv) > 3:
        OUTPUT_DIR = sys.argv[3]
    
    print("\n" + "🔬"*35)
    print("MODEL VALIDATION TOOL")
    print("🔬"*35)
    print("\nUsage: python validate_model.py [model_path] [test_data_dir] [output_dir]")
    print()
    
    results = validate_model(MODEL_PATH, TEST_DATA_DIR, OUTPUT_DIR)
    
    if results:
        print(f"\n✅ Validation complete!")
        print(f"   Plots saved to: {OUTPUT_DIR}")
    
    return results


if __name__ == "__main__":
    main()
