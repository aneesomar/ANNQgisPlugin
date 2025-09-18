# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SimpleTrainingModule
 Simplified training module that works with CSV data
 Fallback when complex QGIS/raster processing is not available
 ***************************************************************************/
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def simple_train_model_from_csv(landslide_csv_path, non_landslide_csv_path, 
                               output_model_path, epochs=150, batch_size=64, 
                               test_split=0.2, progress_callback=None):
    """
    Train model directly from CSV files (simplified workflow)
    
    Args:
        landslide_csv_path: Path to landslide points CSV
        non_landslide_csv_path: Path to non-landslide points CSV  
        output_model_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        test_split: Fraction for test split
        progress_callback: Function to report progress
    """
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
        
    # Safe device selection
    try:
        if torch.cuda.is_available():
            # Test CUDA compatibility
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device('cuda')
            print("Using GPU for training")
        else:
            device = torch.device('cpu')
            print("Using CPU for training")
    except Exception as e:
        print(f"GPU not available ({e}), using CPU")
        device = torch.device('cpu')
        
    if progress_callback:
        progress_callback(0, "Loading data...")
        
    # Load CSV data
    landslides = pd.read_csv(landslide_csv_path)
    non_landslides = pd.read_csv(non_landslide_csv_path)
    
    # Add labels
    landslides["label"] = 1
    non_landslides["label"] = 0
    
    # Combine data
    full_data = pd.concat([landslides, non_landslides], ignore_index=True)
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if progress_callback:
        progress_callback(10, "Preparing features...")
        
    # Separate features and labels
    X = full_data.drop("label", axis=1)
    y = full_data["label"]
    
    # Convert and clean data
    X = X.replace({True: 1, False: 0})
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    
    # Drop coordinate columns
    X = X.drop(columns=["xcoord", "ycoord", "fid"], errors="ignore")
    
    if progress_callback:
        progress_callback(20, "Selecting features...")
        
    # Feature selection
    print(f"Number of features before selection: {X.shape[1]}")
    
    # Statistical feature selection
    selector_stats = SelectKBest(score_func=f_classif, k=min(60, X.shape[1]))
    X_stats_selected = selector_stats.fit_transform(X, y)
    stats_features = X.columns[selector_stats.get_support()]
    
    # Tree-based feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    rf_top_features = feature_importance_rf.head(60).index
    
    # Combine methods
    all_selected = set(stats_features) | set(rf_top_features)
    feature_votes = {}
    for feature in all_selected:
        votes = 0
        if feature in stats_features: votes += 1
        if feature in rf_top_features: votes += 1
        feature_votes[feature] = votes
        
    # Select features with at least 1 vote (more lenient for smaller datasets)
    final_features = [f for f, votes in feature_votes.items() if votes >= 1]
    print(f"Features selected: {len(final_features)}")
    
    X_selected = X[final_features]
    selected_features = final_features
    
    if progress_callback:
        progress_callback(30, "Splitting and scaling data...")
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_split, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    if progress_callback:
        progress_callback(40, "Preparing training...")
        
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Create weighted sampler
    y_train_np = y_train.values
    sample_weights = [class_weight_dict[label] for label in y_train_np]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    if progress_callback:
        progress_callback(50, "Training model...")
        
    # Create simplified model
    model = SimpleLandslideANN(X_train_scaled.shape[1])
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    # Use CPU to avoid CUDA issues
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # Progress reporting
        if progress_callback and epoch % 10 == 0:
            progress = 50 + int((epoch / epochs) * 40)
            progress_callback(progress, f"Training epoch {epoch}/{epochs}")
            
    # Load best model
    model.load_state_dict(best_model_state)
    
    if progress_callback:
        progress_callback(90, "Evaluating model...")
        
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        outputs = model(X_test_tensor)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > 0.5).int().cpu().numpy()
        true = y_test_tensor.int().cpu().numpy()
        
        acc = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, zero_division=0)
        recall = recall_score(true, predicted, zero_division=0)
        f1 = f1_score(true, predicted, zero_division=0)
        
        print(f"\nModel Evaluation:")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
    if progress_callback:
        progress_callback(95, "Saving model...")
        
    # Save model
    output_dir = os.path.dirname(output_model_path)
    if output_dir:  # Only create directory if there is a directory part
        os.makedirs(output_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'selected_features': selected_features,
        'best_threshold': 0.5,
        'model_architecture': 'SimpleLandslideANN',
        'feature_selection_method': 'ensemble',
        'class_weights': class_weight_dict,
        'input_dim': X_train_scaled.shape[1],
        'training_info': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'test_accuracy': acc,
            'test_f1': f1
        }
    }, output_model_path)
    
    if progress_callback:
        progress_callback(100, "Training completed!")
        
    print(f"Model saved to: {output_model_path}")
    return output_model_path

class SimpleLandslideANN(nn.Module):
    """Simplified ANN model for landslide susceptibility"""
    def __init__(self, input_dim):
        super(SimpleLandslideANN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced data"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    # Example usage
    from raster_data_extractor import create_sample_data
    
    # Create sample data
    landslide_path, non_landslide_path = create_sample_data()
    
    # Train model
    output_path = "simple_trained_model.pth"
    
    def progress_print(progress, message):
        print(f"{progress}%: {message}")
        
    simple_train_model_from_csv(
        landslide_path, 
        non_landslide_path, 
        output_path,
        epochs=50,  # Reduced for testing
        progress_callback=progress_print
    )
