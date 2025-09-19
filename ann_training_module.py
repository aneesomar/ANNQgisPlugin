# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ANNTrainingModule
 Core module for training ANN landslide susceptibility models
 Adapted from modelTraining.py to work with QGIS raster and vector data
 ***************************************************************************/
"""

import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from collections import Counter

try:
    from qgis.core import (QgsVectorLayer, QgsRasterLayer, QgsProject, 
                          QgsPointXY, QgsGeometry, QgsFeature, QgsFields, 
                          QgsField, QgsVectorFileWriter, QgsSpatialIndex,
                          QgsCoordinateReferenceSystem, QgsCoordinateTransform)
    from qgis.PyQt.QtCore import QVariant
    import processing
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

class AttentionLayer(nn.Module):
    """Attention mechanism for neural network"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    """Residual block for neural network"""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # Residual connection
        return self.relu(out)

class AdvancedLandslideANN(nn.Module):
    """Advanced ANN model for landslide susceptibility"""
    def __init__(self, input_dim):
        super(AdvancedLandslideANN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(512)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(512, 256, 0.3)
        self.res_block2 = ResidualBlock(512, 256, 0.3)
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(512, 256),
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
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.attention(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.feature_layers(x)
        return self.output(x)

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

class ANNTrainingModule:
    """Main training module for ANN landslide susceptibility models"""
    
    def __init__(self):
        # Force CPU usage for maximum compatibility
        self.device = torch.device('cpu')
        print("Using CPU (forced for compatibility)")
        
    def extract_features_from_rasters(self, raster_paths, landslide_points_path, 
                                     generate_non_landslides=True, progress_callback=None):
        """
        Extract features from rasters at landslide and non-landslide points
        
        Args:
            raster_paths: List of paths to raster files
            landslide_points_path: Path to landslide points vector file
            generate_non_landslides: Whether to generate non-landslide points
            progress_callback: Function to report progress
            
        Returns:
            DataFrame with extracted features
        """
        
        if not QGIS_AVAILABLE:
            raise ImportError("QGIS is required for raster feature extraction")
            
        if progress_callback:
            progress_callback(0)
            
        # Load landslide points
        landslide_layer = QgsVectorLayer(landslide_points_path, "landslides", "ogr")
        if not landslide_layer.isValid():
            raise ValueError(f"Cannot load landslide points from {landslide_points_path}")
            
        # Get CRS from first raster
        first_raster = QgsRasterLayer(raster_paths[0], "temp", "gdal")
        if not first_raster.isValid():
            raise ValueError(f"Cannot load raster: {raster_paths[0]}")
            
        target_crs = first_raster.crs()
        
        # Transform landslide points to raster CRS if needed
        if landslide_layer.crs() != target_crs:
            transform = QgsCoordinateTransform(landslide_layer.crs(), target_crs, QgsProject.instance())
        else:
            transform = None
            
        # Extract landslide point coordinates
        landslide_points = []
        for feature in landslide_layer.getFeatures():
            point = feature.geometry().asPoint()
            if transform:
                point = transform.transform(point)
            landslide_points.append((point.x(), point.y()))
            
        if progress_callback:
            progress_callback(10)
            
        # Generate non-landslide points if requested
        non_landslide_points = []
        if generate_non_landslides:
            non_landslide_points = self._generate_non_landslide_points(
                first_raster, landslide_points, len(landslide_points) * 2
            )
            
        if progress_callback:
            progress_callback(20)
            
        # Combine all points
        all_points = [(x, y, 1) for x, y in landslide_points]  # Landslides = 1
        all_points.extend([(x, y, 0) for x, y in non_landslide_points])  # Non-landslides = 0
        
        # Extract features from all rasters
        features_data = []
        total_rasters = len(raster_paths)
        
        for i, raster_path in enumerate(raster_paths):
            raster_layer = QgsRasterLayer(raster_path, f"raster_{i}", "gdal")
            if not raster_layer.isValid():
                continue
                
            raster_name = os.path.basename(raster_path).split('.')[0]
            
            # Sample raster values at point locations
            for j, (x, y, label) in enumerate(all_points):
                if len(features_data) <= j:
                    features_data.append({'x': x, 'y': y, 'label': label})
                    
                # Sample raster value
                point = QgsPointXY(x, y)
                sample_value = raster_layer.dataProvider().sample(point, 1)[0]
                
                # Handle NoData values
                if sample_value is None or sample_value == raster_layer.dataProvider().sourceNoDataValue(1):
                    sample_value = 0
                    
                features_data[j][raster_name] = sample_value
                
            # Update progress
            if progress_callback:
                progress = 20 + int(((i + 1) / total_rasters) * 60)
                progress_callback(progress)
                
        # Convert to DataFrame
        df = pd.DataFrame(features_data)
        
        if progress_callback:
            progress_callback(100)
            
        return df
        
    def _generate_non_landslide_points(self, reference_raster, landslide_points, num_points):
        """Generate random non-landslide points within raster extent"""
        
        extent = reference_raster.extent()
        
        # Create buffer around landslide points to avoid generating points too close
        min_distance = 100  # meters
        
        non_landslide_points = []
        attempts = 0
        max_attempts = num_points * 10
        
        while len(non_landslide_points) < num_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point within extent
            x = random.uniform(extent.xMinimum(), extent.xMaximum())
            y = random.uniform(extent.yMinimum(), extent.yMaximum())
            
            # Check minimum distance from landslide points
            too_close = False
            for lx, ly in landslide_points:
                distance = np.sqrt((x - lx)**2 + (y - ly)**2)
                if distance < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                # Check if point is within valid raster area (not NoData)
                point = QgsPointXY(x, y)
                sample_value = reference_raster.dataProvider().sample(point, 1)[0]
                
                if (sample_value is not None and 
                    sample_value != reference_raster.dataProvider().sourceNoDataValue(1)):
                    non_landslide_points.append((x, y))
                    
        return non_landslide_points
        
    def prepare_training_data(self, feature_data, test_split=0.2):
        """
        Prepare data for training with feature selection and scaling
        
        Args:
            feature_data: DataFrame with extracted features
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (X, y, selected_features, scaler)
        """
        
        # Separate features and labels
        X = feature_data.drop(['x', 'y', 'label'], axis=1, errors='ignore')
        y = feature_data['label']
        
        # Convert to numeric and handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Feature selection using ensemble method
        selected_features = self._select_features(X, y)
        X_selected = X[selected_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_split, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Combine back into full datasets
        X_scaled = np.vstack([X_train_scaled, X_test_scaled])
        y_combined = pd.concat([y_train, y_test])
        
        return X_scaled, y_combined.values, selected_features, scaler
        
    def _select_features(self, X, y, max_features=60):
        """Select best features using ensemble method"""
        
        print(f"Number of features before selection: {X.shape[1]}")
        
        # Method 1: Statistical (F-test)
        selector_stats = SelectKBest(score_func=f_classif, k=min(max_features, X.shape[1]))
        X_stats_selected = selector_stats.fit_transform(X, y)
        stats_features = X.columns[selector_stats.get_support()]
        
        # Method 2: Tree-based feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        rf_top_features = feature_importance_rf.head(max_features).index
        
        # Method 3: Recursive Feature Elimination
        rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=max_features)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_]
        
        # Combine methods - features that appear in at least 2 out of 3 methods
        all_selected = set(stats_features) | set(rf_top_features) | set(rfe_features)
        feature_votes = {}
        for feature in all_selected:
            votes = 0
            if feature in stats_features: votes += 1
            if feature in rf_top_features: votes += 1
            if feature in rfe_features: votes += 1
            feature_votes[feature] = votes
            
        # Select features with at least 2 votes
        final_features = [f for f, votes in feature_votes.items() if votes >= 2]
        
        print(f"Features selected by ensemble method: {len(final_features)}")
        
        return final_features
        
    def train_model(self, X, y, epochs=150, batch_size=64, progress_callback=None):
        """
        Train the ANN model
        
        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            progress_callback: Function to report progress
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Create train/test split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, stratify=y, random_state=42
        )
        
        # Calculate class weights for imbalanced learning
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Create weighted sampler
        y_train_np = y_train.squeeze().numpy()
        sample_weights = [class_weight_dict[int(label)] for label in y_train_np]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        model = AdvancedLandslideANN(X.shape[1]).to(self.device)
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
        
        # Mixed precision for faster training
        scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                if scaler_amp is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
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
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    if scaler_amp is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                    else:
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
                    
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, epochs)
                
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Training info
        training_info = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'class_weights': class_weight_dict,
            'device': str(self.device)
        }
        
        return model, training_info
        
    def save_model(self, model, scaler, selected_features, training_info, output_path):
        """Save the trained model and associated data"""
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save complete model package
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'selected_features': selected_features,
            'training_info': training_info,
            'model_architecture': 'AdvancedLandslideANN',
            'input_dim': len(selected_features),
        }, output_path)
        
        print(f"Model saved to: {output_path}")
        
    def evaluate_model_performance(self, model, feature_data, scaler, selected_features):
        """
        Evaluate model performance on test set
        
        Args:
            model: Trained PyTorch model
            feature_data: Original DataFrame with extracted features
            scaler: Fitted scaler
            selected_features: List of selected feature names
            
        Returns:
            Dictionary containing performance metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        import torch
        import numpy as np
        import pandas as pd
        
        # Recreate the same data processing as in prepare_training_data
        X = feature_data.drop(['x', 'y', 'label'], axis=1, errors='ignore')
        y = feature_data['label']
        
        # Convert to numeric and handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Select features (use column names for DataFrame indexing)
        X_selected = X[selected_features]
        
        # Split data the same way as training (same random state)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
            test_predictions = (test_probs > 0.5).astype(int)
            
        # Calculate metrics
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions)
        recall = recall_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        
        # Classification report
        class_report = classification_report(y_test, test_predictions, target_names=['Non-Landslide', 'Landslide'])
        
        performance_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'test_size': len(y_test),
            'predictions_distribution': {
                'predicted_landslides': int(np.sum(test_predictions)),
                'predicted_non_landslides': int(len(test_predictions) - np.sum(test_predictions)),
                'actual_landslides': int(np.sum(y_test)),
                'actual_non_landslides': int(len(y_test) - np.sum(y_test))
            }
        }
        
        return performance_metrics
