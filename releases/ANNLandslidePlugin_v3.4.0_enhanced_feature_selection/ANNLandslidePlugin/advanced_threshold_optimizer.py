"""
Advanced Threshold Optimization for Landslide Susceptibility Model
================================================================

This module provides comprehensive threshold optimization using multiple strategies:
1. ROC curve analysis for optimal threshold selection
2. Precision-Recall curve optimization
3. F-beta score optimization (emphasizing recall for landslides)
4. Cost-sensitive threshold selection
5. Landslide-specific optimization using real data

Author: Generated for ANNLandslide Plugin
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedThresholdOptimizer:
    """
    Advanced threshold optimization for landslide susceptibility prediction.
    
    Uses multiple optimization strategies to find the best threshold for different objectives.
    """
    
    def __init__(self, model_path, scaler_path=None, device='cpu'):
        """
        Initialize the threshold optimizer.
        
        Args:
            model_path: Path to the trained model (.pth file)
            scaler_path: Path to the scaler (.pkl file)
            device: Device for model inference ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.device = device
        
        # Load model and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler() if scaler_path else None
        
        # Results storage
        self.optimization_results = {}
        self.thresholds_tested = None
        self.y_true = None
        self.y_proba = None
        
    def _load_model(self):
        """Load the trained PyTorch model."""
        try:
            # Define the model architecture (matching saved model structure)
            class ImprovedLandslideModel(nn.Module):
                def __init__(self, input_size=25, hidden_sizes=[256, 128, 64], dropout_rate=0.5):
                    super().__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_sizes:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)
                        ])
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 1))
                    layers.append(nn.Sigmoid())
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get model parameters from checkpoint
            input_size = checkpoint.get('input_size', 25)
            
            # Create model with correct architecture
            model = ImprovedLandslideModel(input_size=input_size)
            
            # Load model state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
                
            # Store additional info from checkpoint
            self.checkpoint_info = {
                'scaler': checkpoint.get('scaler', None),
                'selected_features': checkpoint.get('selected_features', None),
                'best_threshold': checkpoint.get('best_threshold', 0.5),
                'input_size': input_size
            }
            
            model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_scaler(self):
        """Load the MinMaxScaler for data preprocessing."""
        try:
            import pickle
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler loaded successfully from {self.scaler_path}")
            return scaler
        except Exception as e:
            print(f"⚠️ Could not load scaler: {e}")
            return None
    
    def load_validation_data(self, X_val_path, y_val_path):
        """
        Load validation data for threshold optimization.
        
        Args:
            X_val_path: Path to validation features CSV
            y_val_path: Path to validation labels CSV
        """
        try:
            X_val_df = pd.read_csv(X_val_path)
            y_val = pd.read_csv(y_val_path)
            
            # Convert labels to numpy array
            y_val = y_val.values.ravel() if hasattr(y_val, 'values') else y_val.ravel()
            
            print(f"Original validation data shape: {X_val_df.shape}")
            print(f"Available columns: {list(X_val_df.columns[:10])}...")  # First 10 columns
            
            # Use scaler and selected features from checkpoint if available
            if hasattr(self, 'checkpoint_info'):
                # Apply feature selection FIRST if available
                selected_features = self.checkpoint_info.get('selected_features')
                if selected_features is not None:
                    print(f"Applying feature selection: {len(selected_features)} features")
                    print(f"Selected features: {selected_features[:5]}...")  # First 5
                    
                    # Map selected features to validation data column names
                    available_features = list(X_val_df.columns)
                    mapped_features = []
                    
                    for feature in selected_features:
                        # Try exact match first
                        if feature in available_features:
                            mapped_features.append(feature)
                        else:
                            # Try without suffixes
                            clean_feature = feature.replace('_aligned', '').replace('.0', '')
                            
                            # Map common patterns
                            feature_mapping = {
                                'dem_lo19': 'elv',
                                'distance_road': 'roadProx', 
                                'distance_river': 'riverProx',
                                'planCurv': 'planCurv',
                                'profileCurv': 'profCurv',
                                'Slope': 'slope',
                                'Aspect': 'aspect'
                            }
                            
                            if clean_feature in feature_mapping:
                                mapped_name = feature_mapping[clean_feature]
                                if mapped_name in available_features:
                                    mapped_features.append(mapped_name)
                                    continue
                            
                            # Try direct match without suffixes
                            if clean_feature in available_features:
                                mapped_features.append(clean_feature)
                                continue
                                
                            # For lithology and soil, find closest match
                            if 'lithology_' in feature or 'soil_' in feature:
                                base_name = clean_feature.split('_')[0]  # 'lithology' or 'soil'
                                number = clean_feature.split('_')[1] if '_' in clean_feature else ''
                                
                                # Look for exact number match
                                exact_match = f"{base_name}_{number}"
                                if exact_match in available_features:
                                    mapped_features.append(exact_match)
                                    continue
                                
                                # If no exact match, skip this feature
                                print(f"⚠️ Could not map feature: {feature} -> {clean_feature}")
                            else:
                                print(f"⚠️ Could not map feature: {feature}")
                    
                    print(f"Successfully mapped {len(mapped_features)} out of {len(selected_features)} features")
                    
                    if mapped_features:
                        X_val_selected = X_val_df[mapped_features]
                        
                        # Add dummy columns for missing features to match scaler expectations
                        missing_features = [f for f in selected_features if f not in [
                            feature.replace('_aligned', '').replace('.0', '') 
                            for feature in selected_features 
                            if any(mapped in mapped_features for mapped in [
                                feature.replace('_aligned', '').replace('.0', ''),
                                feature.replace('_aligned', '').replace('.0', '').replace('dem_lo19', 'elv'),
                                feature.replace('_aligned', '').replace('.0', '').replace('distance_road', 'roadProx'),
                                feature.replace('_aligned', '').replace('.0', '').replace('distance_river', 'riverProx'),
                                feature.replace('_aligned', '').replace('.0', '').replace('Slope', 'slope'),
                                feature.replace('_aligned', '').replace('.0', '').replace('Aspect', 'aspect'),
                                feature.replace('_aligned', '').replace('.0', '').replace('planCurv', 'planCurv'),
                                feature.replace('_aligned', '').replace('.0', '').replace('profileCurv', 'profCurv')
                            ])
                        ]]
                        
                        # Simple approach: if we're missing features, add zero columns
                        if len(mapped_features) < len(selected_features):
                            missing_count = len(selected_features) - len(mapped_features)
                            print(f"Adding {missing_count} dummy columns for missing features")
                            
                            # Add zero columns for missing features
                            dummy_data = np.zeros((X_val_selected.shape[0], missing_count))
                            dummy_df = pd.DataFrame(dummy_data, 
                                                  columns=[f'missing_feature_{i}' for i in range(missing_count)])
                            X_val_selected = pd.concat([X_val_selected, dummy_df], axis=1)
                        
                        print(f"After feature adjustment: {X_val_selected.shape}")
                    else:
                        print("❌ No features could be mapped, using first 25 features")
                        X_val_selected = X_val_df.iloc[:, :25]
                    
                    # Convert to numpy array
                    X_val = X_val_selected.values
                else:
                    X_val = X_val_df.values
                
                # Then apply scaler from checkpoint
                checkpoint_scaler = self.checkpoint_info.get('scaler')
                if checkpoint_scaler:
                    print("Using scaler from checkpoint")
                    X_val = checkpoint_scaler.transform(X_val)
                elif self.scaler:
                    print("Using external scaler")
                    X_val = self.scaler.transform(X_val)
            else:
                # Fallback to external scaler
                X_val = X_val_df.values
                if self.scaler:
                    X_val = self.scaler.transform(X_val)
            
            print(f"Final validation data shape: {X_val.shape}")
            
            # Get predictions
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val).to(self.device)
                y_proba = self.model(X_tensor).cpu().numpy().ravel()
            
            self.y_true = y_val
            self.y_proba = y_proba
            
            print(f"✅ Validation data loaded: {len(y_val)} samples")
            print(f"   - Positive samples: {np.sum(y_val)} ({100*np.mean(y_val):.1f}%)")
            print(f"   - Prediction range: [{np.min(y_proba):.3f}, {np.max(y_proba):.3f}]")
            
            return X_val, y_val, y_proba
            
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            raise
    
    def optimize_roc_threshold(self, method='youden'):
        """
        Optimize threshold using ROC curve analysis.
        
        Args:
            method: 'youden' (Youden's J statistic) or 'closest_to_corner'
        
        Returns:
            dict: Optimization results
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Load validation data first using load_validation_data()")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        
        if method == 'youden':
            # Youden's J statistic: maximize (TPR - FPR)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[best_idx]
            
        elif method == 'closest_to_corner':
            # Find point closest to top-left corner (0, 1)
            distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
            best_idx = np.argmin(distances)
            optimal_threshold = thresholds[best_idx]
        
        # Calculate metrics at optimal threshold
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        
        results = {
            'method': f'ROC_{method}',
            'optimal_threshold': optimal_threshold,
            'auc_roc': roc_auc_score(self.y_true, self.y_proba),
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
            'tpr': tpr[best_idx],
            'fpr': fpr[best_idx],
            'youden_j': j_scores[best_idx] if method == 'youden' else tpr[best_idx] - fpr[best_idx]
        }
        
        self.optimization_results['roc_' + method] = results
        return results
    
    def optimize_pr_threshold(self, method='f1_max'):
        """
        Optimize threshold using Precision-Recall curve analysis.
        
        Args:
            method: 'f1_max' or 'balanced_pr'
        
        Returns:
            dict: Optimization results
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Load validation data first using load_validation_data()")
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        
        if method == 'f1_max':
            # Find threshold that maximizes F1 score
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]
            
        elif method == 'balanced_pr':
            # Find threshold where precision ≈ recall
            pr_diff = np.abs(precision[:-1] - recall[:-1])
            best_idx = np.argmin(pr_diff)
            optimal_threshold = thresholds[best_idx]
        
        # Calculate metrics at optimal threshold
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        
        results = {
            'method': f'PR_{method}',
            'optimal_threshold': optimal_threshold,
            'avg_precision': average_precision_score(self.y_true, self.y_proba),
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
        }
        
        self.optimization_results['pr_' + method] = results
        return results
    
    def optimize_fbeta_threshold(self, beta_values=[0.5, 1.0, 1.5, 2.0]):
        """
        Optimize threshold for F-beta scores with different beta values.
        
        Args:
            beta_values: List of beta values to test
        
        Returns:
            dict: Results for each beta value
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Load validation data first using load_validation_data()")
        
        results = {}
        
        # Test thresholds from 0.1 to 0.9
        test_thresholds = np.arange(0.1, 0.91, 0.01)
        
        for beta in beta_values:
            best_threshold = 0.5
            best_fbeta = 0
            
            fbeta_scores = []
            for threshold in test_thresholds:
                y_pred = (self.y_proba >= threshold).astype(int)
                fbeta = fbeta_score(self.y_true, y_pred, beta=beta, zero_division=0)
                fbeta_scores.append(fbeta)
                
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_threshold = threshold
            
            # Calculate final metrics
            y_pred = (self.y_proba >= best_threshold).astype(int)
            
            results[f'fbeta_{beta}'] = {
                'method': f'F-beta_{beta}',
                'beta': beta,
                'optimal_threshold': best_threshold,
                'fbeta_score': best_fbeta,
                'accuracy': accuracy_score(self.y_true, y_pred),
                'precision': precision_score(self.y_true, y_pred, zero_division=0),
                'recall': recall_score(self.y_true, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
            }
            
            self.optimization_results[f'fbeta_{beta}'] = results[f'fbeta_{beta}']
        
        return results
    
    def optimize_cost_sensitive_threshold(self, cost_matrix=None):
        """
        Optimize threshold using cost-sensitive approach.
        
        Args:
            cost_matrix: 2x2 matrix [[TN_cost, FP_cost], [FN_cost, TP_cost]]
                        Default: [[0, 1], [10, 0]] (FN costs 10x more than FP)
        
        Returns:
            dict: Optimization results
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Load validation data first using load_validation_data()")
        
        if cost_matrix is None:
            # Default: False negatives (missing landslides) are 10x costlier than false positives
            cost_matrix = np.array([[0, 1],    # [TN_cost, FP_cost]
                                   [10, 0]])   # [FN_cost, TP_cost]
        
        test_thresholds = np.arange(0.05, 0.96, 0.01)
        best_threshold = 0.5
        min_cost = float('inf')
        costs = []
        
        for threshold in test_thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
            
            # Calculate total cost
            total_cost = (tn * cost_matrix[0,0] + fp * cost_matrix[0,1] + 
                         fn * cost_matrix[1,0] + tp * cost_matrix[1,1])
            costs.append(total_cost)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_threshold = threshold
        
        # Calculate final metrics
        y_pred = (self.y_proba >= best_threshold).astype(int)
        
        results = {
            'method': 'Cost_Sensitive',
            'optimal_threshold': best_threshold,
            'min_cost': min_cost,
            'cost_matrix': cost_matrix.tolist(),
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
        }
        
        self.optimization_results['cost_sensitive'] = results
        return results
    
    def optimize_landslide_focused_threshold(self, target_recall=0.8):
        """
        Optimize threshold specifically for landslide detection.
        Prioritizes recall (catching landslides) while maintaining reasonable precision.
        
        Args:
            target_recall: Minimum recall to achieve
        
        Returns:
            dict: Optimization results
        """
        if self.y_true is None or self.y_proba is None:
            raise ValueError("Load validation data first using load_validation_data()")
        
        test_thresholds = np.arange(0.05, 0.96, 0.01)
        valid_thresholds = []
        
        # Find thresholds that achieve target recall
        for threshold in test_thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            
            if recall >= target_recall:
                precision = precision_score(self.y_true, y_pred, zero_division=0)
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                
                valid_thresholds.append({
                    'threshold': threshold,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                })
        
        if not valid_thresholds:
            # If target recall can't be achieved, find best compromise
            best_threshold = 0.5
            best_score = 0
            
            for threshold in test_thresholds:
                y_pred = (self.y_proba >= threshold).astype(int)
                recall = recall_score(self.y_true, y_pred, zero_division=0)
                precision = precision_score(self.y_true, y_pred, zero_division=0)
                
                # Weighted score favoring recall (2:1 ratio)
                score = (2 * recall + precision) / 3
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        else:
            # Among valid thresholds, choose the one with highest precision
            best_result = max(valid_thresholds, key=lambda x: x['precision'])
            best_threshold = best_result['threshold']
        
        # Calculate final metrics
        y_pred = (self.y_proba >= best_threshold).astype(int)
        
        results = {
            'method': f'Landslide_Focused_Recall_{target_recall}',
            'optimal_threshold': best_threshold,
            'target_recall': target_recall,
            'achieved_recall': recall_score(self.y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
        }
        
        self.optimization_results['landslide_focused'] = results
        return results
    
    def run_comprehensive_optimization(self):
        """
        Run all optimization methods and return comprehensive results.
        
        Returns:
            dict: All optimization results
        """
        print("🚀 Running Comprehensive Threshold Optimization...")
        
        # ROC-based optimization
        print("\n📊 ROC Curve Optimization...")
        self.optimize_roc_threshold('youden')
        self.optimize_roc_threshold('closest_to_corner')
        
        # PR-based optimization
        print("📈 Precision-Recall Optimization...")
        self.optimize_pr_threshold('f1_max')
        self.optimize_pr_threshold('balanced_pr')
        
        # F-beta optimization
        print("🎯 F-beta Score Optimization...")
        self.optimize_fbeta_threshold([0.5, 1.0, 1.5, 2.0])
        
        # Cost-sensitive optimization
        print("💰 Cost-Sensitive Optimization...")
        self.optimize_cost_sensitive_threshold()
        
        # Landslide-focused optimization
        print("🏔️ Landslide-Focused Optimization...")
        self.optimize_landslide_focused_threshold(0.8)
        self.optimize_landslide_focused_threshold(0.9)
        
        return self.optimization_results
    
    def create_optimization_report(self, save_path="threshold_optimization_report.html"):
        """
        Create a comprehensive HTML report of all optimization results.
        
        Args:
            save_path: Path to save the HTML report
        """
        if not self.optimization_results:
            print("❌ No optimization results found. Run optimization first.")
            return
        
        # Create visualizations
        self.plot_threshold_analysis()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Optimization report saved to: {save_path}")
    
    def plot_threshold_analysis(self):
        """Create comprehensive visualization plots for threshold analysis."""
        if self.y_true is None or self.y_proba is None:
            print("❌ No data available for plotting")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROC Curve with optimal thresholds
        plt.subplot(2, 3, 1)
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_proba)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(self.y_true, self.y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Mark optimal points
        if 'roc_youden' in self.optimization_results:
            result = self.optimization_results['roc_youden']
            plt.plot(result['fpr'], result['tpr'], 'ro', markersize=10, label=f"Youden's J (t={result['optimal_threshold']:.3f})")
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Optimal Thresholds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision, recall, pr_thresholds = precision_recall_curve(self.y_true, self.y_proba)
        plt.plot(recall, precision, 'g-', linewidth=2, 
                label=f'PR Curve (AP = {average_precision_score(self.y_true, self.y_proba):.3f})')
        
        # Mark optimal points
        if 'pr_f1_max' in self.optimization_results:
            result = self.optimization_results['pr_f1_max']
            plt.plot(result['recall'], result['precision'], 'ro', markersize=10, 
                    label=f"Max F1 (t={result['optimal_threshold']:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Threshold vs Metrics
        plt.subplot(2, 3, 3)
        test_thresholds = np.arange(0.1, 0.91, 0.02)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for threshold in test_thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(self.y_true, y_pred, zero_division=0))
            precision_scores.append(precision_score(self.y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(self.y_true, y_pred, zero_division=0))
        
        plt.plot(test_thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
        plt.plot(test_thresholds, precision_scores, 'r-', linewidth=2, label='Precision')
        plt.plot(test_thresholds, recall_scores, 'g-', linewidth=2, label='Recall')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.y_proba[self.y_true == 0], bins=50, alpha=0.7, label='Non-landslide', color='blue', density=True)
        plt.hist(self.y_proba[self.y_true == 1], bins=50, alpha=0.7, label='Landslide', color='red', density=True)
        
        # Mark optimal thresholds
        for method, result in self.optimization_results.items():
            if method.startswith('roc_youden'):
                plt.axvline(result['optimal_threshold'], color='green', linestyle='--', 
                           label=f"ROC Optimal: {result['optimal_threshold']:.3f}")
                break
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Confusion Matrix Heatmap (for best threshold)
        plt.subplot(2, 3, 5)
        best_method = max(self.optimization_results.items(), key=lambda x: x[1].get('f1_score', 0))
        best_threshold = best_method[1]['optimal_threshold']
        y_pred = (self.y_proba >= best_threshold).astype(int)
        
        cm = confusion_matrix(self.y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-landslide', 'Landslide'],
                   yticklabels=['Non-landslide', 'Landslide'])
        plt.title(f'Confusion Matrix\n{best_method[0]} (t={best_threshold:.3f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 6. F-beta Scores Comparison
        plt.subplot(2, 3, 6)
        fbeta_methods = [(k, v) for k, v in self.optimization_results.items() if k.startswith('fbeta_')]
        if fbeta_methods:
            beta_values = [float(k.split('_')[1]) for k, _ in fbeta_methods]
            fbeta_scores = [v['fbeta_score'] for _, v in fbeta_methods]
            thresholds = [v['optimal_threshold'] for _, v in fbeta_methods]
            
            plt.bar(range(len(beta_values)), fbeta_scores, color='skyblue', alpha=0.7)
            plt.xlabel('Beta Value')
            plt.ylabel('F-beta Score')
            plt.title('F-beta Optimization Results')
            plt.xticks(range(len(beta_values)), [f'β={b}' for b in beta_values])
            
            # Add threshold labels
            for i, (score, threshold) in enumerate(zip(fbeta_scores, thresholds)):
                plt.text(i, score + 0.01, f't={threshold:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Threshold analysis plots saved to: threshold_optimization_analysis.png")
        plt.show()
    
    def _generate_html_report(self):
        """Generate HTML report content."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Threshold Optimization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 10px; }
                .method { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .best { background-color: #e8f5e8; border-color: #4CAF50; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
                .threshold { color: #2196F3; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Advanced Threshold Optimization Report</h1>
                <p><strong>Generated:</strong> October 13, 2025</p>
                <p><strong>Model:</strong> Improved Landslide Susceptibility Model</p>
                <p><strong>Validation Samples:</strong> """ + str(len(self.y_true)) + """</p>
            </div>
        """
        
        # Add results summary table
        html_content += """
        <h2>📊 Optimization Results Summary</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Threshold</th>
                <th>F1 Score</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Accuracy</th>
            </tr>
        """
        
        # Sort results by F1 score
        sorted_results = sorted(self.optimization_results.items(), 
                              key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        for i, (method, result) in enumerate(sorted_results):
            best_class = ' class="best"' if i == 0 else ''
            html_content += f"""
            <tr{best_class}>
                <td>{result['method']}</td>
                <td class="threshold">{result['optimal_threshold']:.3f}</td>
                <td class="metric">{result.get('f1_score', 0):.3f}</td>
                <td>{result.get('precision', 0):.3f}</td>
                <td>{result.get('recall', 0):.3f}</td>
                <td>{result.get('accuracy', 0):.3f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        
        <h2>🔍 Detailed Method Analysis</h2>
        """
        
        # Add detailed results for each method
        for method, result in sorted_results:
            html_content += f"""
            <div class="method">
                <h3>{result['method']}</h3>
                <p><strong>Optimal Threshold:</strong> <span class="threshold">{result['optimal_threshold']:.3f}</span></p>
                <p><strong>Key Metrics:</strong></p>
                <ul>
                    <li>F1 Score: {result.get('f1_score', 0):.3f}</li>
                    <li>Precision: {result.get('precision', 0):.3f}</li>
                    <li>Recall: {result.get('recall', 0):.3f}</li>
                    <li>Accuracy: {result.get('accuracy', 0):.3f}</li>
                </ul>
            </div>
            """
        
        html_content += """
        <h2>📈 Visualizations</h2>
        <p>Comprehensive analysis plots have been saved as: <strong>threshold_optimization_analysis.png</strong></p>
        
        <h2>💡 Recommendations</h2>
        <div class="method">
            <h3>🎯 Best Overall Performance</h3>
            <p>Based on F1 score, the optimal method is: <strong>""" + sorted_results[0][1]['method'] + """</strong></p>
            <p>Recommended threshold: <span class="threshold">""" + f"{sorted_results[0][1]['optimal_threshold']:.3f}" + """</span></p>
        </div>
        
        </body>
        </html>
        """
        
        return html_content
    
    def get_best_threshold(self, criterion='f1_score'):
        """
        Get the best threshold based on specified criterion.
        
        Args:
            criterion: 'f1_score', 'recall', 'precision', or 'accuracy'
        
        Returns:
            tuple: (best_threshold, method_name, metrics_dict)
        """
        if not self.optimization_results:
            raise ValueError("No optimization results found. Run optimization first.")
        
        best_method = max(self.optimization_results.items(), 
                         key=lambda x: x[1].get(criterion, 0))
        
        return (
            best_method[1]['optimal_threshold'],
            best_method[0],
            best_method[1]
        )


def main():
    """
    Main function to run threshold optimization.
    """
    print("🚀 Advanced Threshold Optimization for Landslide Susceptibility")
    print("=" * 60)
    
    # Configuration
    model_path = "outputs/output2.pth"  # Using improved model
    scaler_path = "models/minmax_scaler_original.pkl"
    X_val_path = "ANN-landslide-susceptibility/data/X_val.csv"
    y_val_path = "ANN-landslide-susceptibility/data/y_val.csv"
    
    try:
        # Initialize optimizer
        print("🔧 Initializing threshold optimizer...")
        optimizer = AdvancedThresholdOptimizer(
            model_path=model_path,
            scaler_path=scaler_path,
            device='cpu'
        )
        
        # Load validation data
        print("📂 Loading validation data...")
        optimizer.load_validation_data(X_val_path, y_val_path)
        
        # Run comprehensive optimization
        results = optimizer.run_comprehensive_optimization()
        
        # Create report
        print("\n📊 Generating comprehensive report...")
        optimizer.create_optimization_report("advanced_threshold_optimization_report.html")
        
        # Print best results
        print("\n" + "="*60)
        print("🏆 BEST THRESHOLD RECOMMENDATIONS")
        print("="*60)
        
        for criterion in ['f1_score', 'recall', 'precision']:
            best_threshold, method, metrics = optimizer.get_best_threshold(criterion)
            print(f"\n🎯 Best for {criterion.upper()}:")
            print(f"   Method: {method}")
            print(f"   Threshold: {best_threshold:.3f}")
            print(f"   F1: {metrics['f1_score']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
        
        print("\n✅ Threshold optimization completed successfully!")
        print("📊 Check 'advanced_threshold_optimization_report.html' for detailed results")
        print("📈 Check 'threshold_optimization_analysis.png' for visualizations")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()