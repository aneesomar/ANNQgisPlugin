# -*- coding: utf-8 -*-
"""
/***************************************************************************
 CSV-Only Training Module
 Simple training module that only works with CSV files and has minimal dependencies
 ***************************************************************************/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SimplestLandslideANN(nn.Module):
    """Very simple ANN model that avoids CUDA issues"""
    def __init__(self, input_dim):
        super(SimplestLandslideANN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def csv_only_training(landslide_csv_path, non_landslide_csv_path, 
                     output_model_path, epochs=50, progress_callback=None):
    """
    Simple CSV-only training that avoids all compatibility issues
    
    Args:
        landslide_csv_path: Path to landslide CSV
        non_landslide_csv_path: Path to non-landslide CSV
        output_model_path: Path to save model
        epochs: Number of training epochs
        progress_callback: Function to report progress
    """
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
        
    if progress_callback:
        progress_callback(0, "Loading CSV data...")
        
    # Load data
    try:
        landslides = pd.read_csv(landslide_csv_path)
        non_landslides = pd.read_csv(non_landslide_csv_path)
    except Exception as e:
        raise ValueError(f"Cannot load CSV files: {e}")
        
    # Add labels
    landslides["label"] = 1
    non_landslides["label"] = 0
    
    # Combine and shuffle
    full_data = pd.concat([landslides, non_landslides], ignore_index=True)
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if progress_callback:
        progress_callback(10, "Preparing features...")
        
    # Prepare features
    X = full_data.drop("label", axis=1)
    y = full_data["label"]
    
    # Clean data
    X = X.replace({True: 1, False: 0})
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    
    # Remove coordinate columns
    coordinate_cols = ["xcoord", "ycoord", "fid", "x", "y"]
    for col in coordinate_cols:
        if col in X.columns:
            X = X.drop(col, axis=1)
            
    if progress_callback:
        progress_callback(20, "Selecting features...")
        
    # Simple feature selection
    if X.shape[1] > 10:
        selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
        X = pd.DataFrame(selector.fit_transform(X, y), 
                        columns=X.columns[selector.get_support()])
        
    selected_features = X.columns.tolist()
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    if progress_callback:
        progress_callback(30, "Splitting and scaling data...")
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Force CPU usage to avoid CUDA issues
    device = torch.device('cpu')
    print("Using CPU for training (avoiding CUDA issues)")
    
    if progress_callback:
        progress_callback(40, "Converting to tensors...")
        
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    if progress_callback:
        progress_callback(50, "Setting up training...")
        
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create simple model
    model = SimplestLandslideANN(X_train_scaled.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if progress_callback:
        progress_callback(60, "Training model...")
        
    # Simple training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if progress_callback and epoch % 10 == 0:
            progress = 60 + int((epoch / epochs) * 30)
            progress_callback(progress, f"Training epoch {epoch+1}/{epochs}")
            
    if progress_callback:
        progress_callback(90, "Evaluating model...")
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > 0.5).int().numpy()
        true = y_test_tensor.int().numpy()
        
        acc = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, zero_division=0)
        recall = recall_score(true, predicted, zero_division=0)
        f1 = f1_score(true, predicted, zero_division=0)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
    if progress_callback:
        progress_callback(95, "Saving model...")
        
    # Save model
    output_dir = os.path.dirname(output_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'selected_features': selected_features,
        'model_architecture': 'SimplestLandslideANN',
        'input_dim': X_train_scaled.shape[1],
        'training_info': {
            'epochs_trained': epochs,
            'test_accuracy': acc,
            'test_f1': f1,
            'device': 'cpu'
        }
    }, output_model_path)
    
    if progress_callback:
        progress_callback(100, "Training completed!")
        
    print(f"Model saved to: {output_model_path}")
    return output_model_path

def create_simple_sample_data(output_dir="."):
    """Create simple sample data without any complex dependencies"""
    
    # Simple feature generation
    np.random.seed(42)
    
    # Features
    features = ['slope', 'elevation', 'aspect', 'distance_river', 'distance_road', 
               'geology', 'soil', 'curvature']
    
    # Landslide data (higher risk characteristics)
    n_landslides = 100
    landslide_data = {
        'slope': np.random.normal(30, 10, n_landslides),  # Steeper slopes
        'elevation': np.random.normal(1200, 300, n_landslides),
        'aspect': np.random.uniform(0, 360, n_landslides),
        'distance_river': np.random.exponential(200, n_landslides),  # Closer to rivers
        'distance_road': np.random.exponential(300, n_landslides),
        'geology': np.random.randint(1, 5, n_landslides),
        'soil': np.random.randint(1, 4, n_landslides),
        'curvature': np.random.normal(0, 0.1, n_landslides)
    }
    
    # Non-landslide data (lower risk characteristics)
    n_non_landslides = 200
    non_landslide_data = {
        'slope': np.random.normal(15, 8, n_non_landslides),  # Gentler slopes
        'elevation': np.random.normal(800, 400, n_non_landslides),
        'aspect': np.random.uniform(0, 360, n_non_landslides),
        'distance_river': np.random.exponential(500, n_non_landslides),  # Further from rivers
        'distance_road': np.random.exponential(600, n_non_landslides),
        'geology': np.random.randint(1, 5, n_non_landslides),
        'soil': np.random.randint(1, 4, n_non_landslides),
        'curvature': np.random.normal(0, 0.05, n_non_landslides)
    }
    
    # Add dummy coordinates
    landslide_data['xcoord'] = np.random.uniform(28, 31, n_landslides)
    landslide_data['ycoord'] = np.random.uniform(-30, -29, n_landslides)
    landslide_data['fid'] = range(n_landslides)
    
    non_landslide_data['xcoord'] = np.random.uniform(28, 31, n_non_landslides)
    non_landslide_data['ycoord'] = np.random.uniform(-30, -29, n_non_landslides)
    non_landslide_data['fid'] = range(n_non_landslides)
    
    # Save to CSV
    landslide_df = pd.DataFrame(landslide_data)
    non_landslide_df = pd.DataFrame(non_landslide_data)
    
    landslide_path = os.path.join(output_dir, 'simple_landslides.csv')
    non_landslide_path = os.path.join(output_dir, 'simple_non_landslides.csv')
    
    landslide_df.to_csv(landslide_path, index=False)
    non_landslide_df.to_csv(non_landslide_path, index=False)
    
    print(f"Simple sample data created:")
    print(f"  Landslides: {landslide_path}")
    print(f"  Non-landslides: {non_landslide_path}")
    
    return landslide_path, non_landslide_path

if __name__ == "__main__":
    # Test the simple training
    print("Creating simple sample data...")
    landslide_path, non_landslide_path = create_simple_sample_data()
    
    print("Training simple model...")
    
    def progress_print(progress, message):
        print(f"{progress}%: {message}")
        
    try:
        model_path = csv_only_training(
            landslide_path, 
            non_landslide_path, 
            "simple_test_model.pth",
            epochs=30,
            progress_callback=progress_print
        )
        print(f"✅ Simple training successful! Model: {model_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()