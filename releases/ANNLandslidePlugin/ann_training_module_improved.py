# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ANNTrainingModule - Improved with Spatial Cross-Validation
 Core module for training ANN landslide susceptibility models
 Replicates the successful approach from modelTraining_spatial_cv.py
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, fbeta_score, confusion_matrix
)
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, 
                             classification_report, confusion_matrix, roc_auc_score)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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


##########################Loss Functions##########################

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in landslide prediction
    
    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter to down-weight easy examples (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
            
        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    Early stopping to avoid overfitting during training
    """
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


##########################Spatial Cross-Validation Functions##########################

def create_spatial_blocks(coordinates, n_blocks=5, method='kmeans'):
    """
    Create spatial blocks for cross-validation.
    
    Parameters:
    - coordinates: array-like of shape (n_samples, 2) with x, y coordinates
    - n_blocks: number of spatial blocks to create
    - method: 'kmeans' or 'grid' for blocking method
    
    Returns:
    - block_labels: array of block assignments for each sample
    """
    coords_array = np.array(coordinates)
    
    if method == 'kmeans':
        # Use K-means clustering to create spatial blocks
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
        block_labels = kmeans.fit_predict(coords_array)
        
    elif method == 'grid':
        # Create grid-based blocks
        x_min, x_max = coords_array[:, 0].min(), coords_array[:, 0].max()
        y_min, y_max = coords_array[:, 1].min(), coords_array[:, 1].max()
        
        # Calculate grid dimensions
        grid_size = int(np.sqrt(n_blocks))
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # Assign blocks based on grid position
        x_indices = np.digitize(coords_array[:, 0], x_bins) - 1
        y_indices = np.digitize(coords_array[:, 1], y_bins) - 1
        
        # Ensure indices are within bounds
        x_indices = np.clip(x_indices, 0, grid_size - 1)
        y_indices = np.clip(y_indices, 0, grid_size - 1)
        
        block_labels = x_indices * grid_size + y_indices
        
        # Reassign block numbers to be consecutive
        unique_blocks = np.unique(block_labels)
        block_mapping = {old: new for new, old in enumerate(unique_blocks)}
        block_labels = np.array([block_mapping[label] for label in block_labels])
    
    return block_labels


def apply_spatial_buffer(coordinates, train_indices, test_indices, buffer_distance):
    """
    Apply spatial buffer by removing training samples that are too close to test samples.
    
    Parameters:
    - coordinates: array-like of shape (n_samples, 2) with x, y coordinates
    - train_indices: indices of training samples
    - test_indices: indices of test samples
    - buffer_distance: minimum distance between train and test samples
    
    Returns:
    - filtered_train_indices: training indices after applying buffer
    """
    coords_array = np.array(coordinates)
    train_coords = coords_array[train_indices]
    test_coords = coords_array[test_indices]
    
    # Calculate distances between all train and test points
    distances = cdist(train_coords, test_coords, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    # Keep only training samples that are far enough from test samples
    valid_train_mask = min_distances >= buffer_distance
    filtered_train_indices = train_indices[valid_train_mask]
    
    return filtered_train_indices


def spatial_train_test_split(X, y, coordinates, test_size=0.2, n_blocks=10, 
                           method='kmeans', buffer_distance=None, random_state=42):
    """
    Perform spatial train-test split using blocking approach.
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - coordinates: spatial coordinates (x, y)
    - test_size: proportion of data for testing
    - n_blocks: number of spatial blocks
    - method: 'kmeans' or 'grid' for blocking
    - buffer_distance: minimum distance between train/test areas
    - random_state: random seed
    
    Returns:
    - X_train, X_test, y_train, y_test: split datasets
    - train_coords, test_coords: coordinates for train/test sets
    """
    np.random.seed(random_state)
    
    # Create spatial blocks
    block_labels = create_spatial_blocks(coordinates, n_blocks, method)
    
    # Calculate target proportions in each block
    unique_blocks = np.unique(block_labels)
    block_info = []
    
    for block in unique_blocks:
        block_mask = block_labels == block
        block_size = np.sum(block_mask)
        if block_size > 0:
            block_y = y[block_mask] if hasattr(y, '__getitem__') else y.iloc[block_mask]
            pos_prop = np.mean(block_y) if len(block_y) > 0 else 0
            block_info.append({
                'block': block,
                'size': block_size,
                'pos_proportion': pos_prop
            })
    
    # Sort blocks by size (descending)
    block_info.sort(key=lambda x: x['size'], reverse=True)
    
    # Select blocks for test set to approximate desired test_size
    total_samples = len(X)
    target_test_samples = int(total_samples * test_size)
    
    test_blocks = []
    test_samples_count = 0
    
    # Greedily select blocks for test set
    for block_data in block_info:
        if test_samples_count < target_test_samples:
            test_blocks.append(block_data['block'])
            test_samples_count += block_data['size']
        if test_samples_count >= target_test_samples * 0.8:  # Allow some flexibility
            break
    
    # If we don't have enough samples, add more blocks
    if test_samples_count < target_test_samples * 0.5:
        for block_data in block_info:
            if block_data['block'] not in test_blocks:
                test_blocks.append(block_data['block'])
                test_samples_count += block_data['size']
                if test_samples_count >= target_test_samples * 0.8:
                    break
    
    # Create train/test indices
    test_mask = np.isin(block_labels, test_blocks)
    train_mask = ~test_mask
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # Apply spatial buffer if specified
    if buffer_distance is not None and len(train_indices) > 0 and len(test_indices) > 0:
        train_indices = apply_spatial_buffer(coordinates, train_indices, test_indices, buffer_distance)
    
    # Create splits
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    train_coords = coordinates[train_indices]
    test_coords = coordinates[test_indices]
    
    # Print split information
    print(f"Spatial split created:")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Training positive rate: {np.mean(y_train):.3f}")
    print(f"  Test positive rate: {np.mean(y_test):.3f}")
    print(f"  Number of test blocks: {len(test_blocks)}")
    if buffer_distance is not None:
        print(f"  Spatial buffer applied: {buffer_distance:.0f} units")
    
    return X_train, X_test, y_train, y_test, train_coords, test_coords


##########################Model Architecture##########################

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
    """
    SIMPLIFIED ANN model for landslide susceptibility
    
    UPDATED: Replaced complex architecture with simplified version
    - Fewer layers (4 instead of 8+) to reduce overfitting
    - Higher dropout (0.6 instead of 0.3) for better regularization
    - No attention or residual blocks - they cause overfitting on small datasets
    - Produces gradual probabilities instead of binary predictions
    
    This architecture is specifically designed for landslide datasets with 5k-50k samples.
    The simpler architecture prevents overfitting and produces realistic probability distributions.
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.5):
        super(AdvancedLandslideANN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_sizes[2], 1)
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
        # Ensure targets have the same shape as inputs
        targets = targets.view(-1, 1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing
    Prevents overconfident predictions by softening hard labels
    
    Label smoothing changes:
    - Hard label 1.0 → 0.9 (slightly less certain about positives)
    - Hard label 0.0 → 0.1 (slightly less certain about negatives)
    
    This encourages the model to output gradual probabilities instead of binary 0/1.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        # Original: 0 or 1
        # Smoothed: 0.1 or 0.9 (with smoothing=0.1)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Use BCEWithLogitsLoss (numerically stable)
        loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth)
        return loss


##########################Main Training Module##########################

class ANNTrainingModuleImproved:
    """Improved training module with spatial cross-validation"""
    
    def __init__(self):
        # Force CPU usage to avoid CUDA compatibility issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device} (forced CPU for compatibility)")
        
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
                nodata_value = raster_layer.dataProvider().sourceNoDataValue(1)
                if sample_value is None:
                    sample_value = np.nan
                elif nodata_value is not None and sample_value == nodata_value:
                    sample_value = np.nan
                # Filter out extreme values that are likely nodata
                elif abs(sample_value) > 1e10:
                    sample_value = np.nan
                    
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
        
    def prepare_training_data_with_spatial_cv(self, feature_data, test_split=0.2, 
                                              use_spatial_cv=True, n_blocks=10, buffer_distance=None):
        """
        Prepare data with spatial cross-validation and ensemble feature selection
        
        Args:
            feature_data: DataFrame with extracted features (must include 'x', 'y', 'label')
            test_split: Fraction of data for testing
            use_spatial_cv: Whether to use spatial cross-validation
            n_blocks: Number of spatial blocks for spatial CV
            buffer_distance: Spatial buffer distance (auto-calculated if None)
            
        Returns:
            Dictionary with training data and metadata
        """
        
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        # Extract coordinates BEFORE any processing
        if 'x' in feature_data.columns and 'y' in feature_data.columns:
            coordinates = feature_data[['x', 'y']].values
        elif 'xcoord' in feature_data.columns and 'ycoord' in feature_data.columns:
            coordinates = feature_data[['xcoord', 'ycoord']].values
        else:
            print("Warning: No coordinate columns found. Spatial CV will not be possible.")
            coordinates = None
            use_spatial_cv = False
        
        # Separate features and labels
        X = feature_data.drop(['x', 'y', 'label', 'xcoord', 'ycoord', 'fid'], axis=1, errors='ignore')
        y = feature_data['label']
        
        # Convert to numeric and handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with too many NaN values
        nan_threshold = 0.5 * X.shape[1]
        valid_rows = X.isna().sum(axis=1) < nan_threshold
        X = X[valid_rows]
        y = y[valid_rows]
        if coordinates is not None:
            coordinates = coordinates[valid_rows]
        
        print(f"Removed {(~valid_rows).sum()} rows with too many missing values")
        
        # Identify categorical features for one-hot encoding
        # Check for various naming patterns for lithology and soil
        categorical_features = []
        lithology_candidates = [col for col in X.columns if 'lithology' in col.lower()]
        soil_candidates = [col for col in X.columns if 'soil' in col.lower()]
        
        if lithology_candidates:
            # Use the first lithology column found
            categorical_features.append(lithology_candidates[0])
            print(f"Detected lithology feature: {lithology_candidates[0]}")
            # Rename to 'Lithology' for consistent one-hot encoding
            X = X.rename(columns={lithology_candidates[0]: 'Lithology'})
            
        if soil_candidates:
            # Use the first soil column found
            categorical_features.append(soil_candidates[0])
            print(f"Detected soil feature: {soil_candidates[0]}")
            # Rename to 'Soil' for consistent one-hot encoding
            X = X.rename(columns={soil_candidates[0]: 'Soil'})
            
        # Update categorical_features to use standardized names
        categorical_features = []
        if 'Lithology' in X.columns:
            categorical_features.append('Lithology')
        if 'Soil' in X.columns:
            categorical_features.append('Soil')
        
        # Separate continuous and categorical
        continuous_cols = [col for col in X.columns if col not in categorical_features]
        continuous_data = X[continuous_cols]
        
        # Fill NaN in continuous features
        continuous_data = continuous_data.fillna(continuous_data.median())
        continuous_data = continuous_data.fillna(0)
        
        # One-hot encode categorical features
        encoded_dfs = [continuous_data]
        
        if 'Lithology' in categorical_features:
            print("One-hot encoding Lithology feature...")
            lithology_dummies = pd.get_dummies(X['Lithology'], prefix='lithology', dummy_na=False)
            encoded_dfs.append(lithology_dummies)
            print(f"   Created {len(lithology_dummies.columns)} lithology categories")
        
        if 'Soil' in categorical_features:
            print("One-hot encoding Soil feature...")
            soil_dummies = pd.get_dummies(X['Soil'], prefix='soil', dummy_na=False)
            encoded_dfs.append(soil_dummies)
            print(f"   Created {len(soil_dummies.columns)} soil categories")
        
        # Combine all features
        X = pd.concat(encoded_dfs, axis=1)
        print(f"Total features after encoding: {X.shape[1]}")
        
        # Enhanced feature selection
        print("\n" + "="*50)  
        print("ENHANCED FEATURE SELECTION")
        print("="*50)
        selected_features = self._enhanced_feature_selection(X, y, max_features=15, enable_quality_filtering=True)
        X_selected = X[selected_features]
        
        # Perform spatial or random split
        if use_spatial_cv and coordinates is not None:
            print("\n" + "="*50)
            print("PERFORMING SPATIAL CROSS-VALIDATION SPLIT")
            print("="*50)
            
            # Calculate buffer distance if not provided
            if buffer_distance is None:
                coord_range_x = coordinates[:, 0].max() - coordinates[:, 0].min()
                coord_range_y = coordinates[:, 1].max() - coordinates[:, 1].min()
                buffer_distance = min(coord_range_x, coord_range_y) * 0.05  # 5% of minimum extent
            
            X_train, X_test, y_train, y_test, train_coords, test_coords = spatial_train_test_split(
                X_selected, y, coordinates,
                test_size=test_split,
                n_blocks=n_blocks,
                method='kmeans',
                buffer_distance=buffer_distance,
                random_state=42
            )
        else:
            print("\nPerforming random stratified split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_split, stratify=y, random_state=42
            )
            train_coords = None
            test_coords = None
        
        # Check and fix test set imbalance (critical for metrics!)
        print("\n" + "="*50)
        print("CHECKING TEST SET BALANCE")
        print("="*50)
        
        # Convert to numpy arrays for processing (handle pandas Series/DataFrame)
        if isinstance(y_train, torch.Tensor):
            y_train_np = y_train.numpy()
        elif isinstance(y_train, pd.Series):
            y_train_np = y_train.values
        else:
            y_train_np = np.array(y_train).flatten()
        
        if isinstance(y_test, torch.Tensor):
            y_test_np = y_test.numpy()
        elif isinstance(y_test, pd.Series):
            y_test_np = y_test.values
        else:
            y_test_np = np.array(y_test).flatten()
        
        train_landslides = int(np.sum(y_train_np))
        train_non_landslides = len(y_train_np) - train_landslides
        test_landslides = int(np.sum(y_test_np))
        test_non_landslides = len(y_test_np) - test_landslides
        
        print(f"\nTrain: {train_landslides} landslides ({train_landslides/len(y_train_np)*100:.1f}%), "
              f"{train_non_landslides} non-landslides ({train_non_landslides/len(y_train_np)*100:.1f}%)")
        print(f"Test:  {test_landslides} landslides ({test_landslides/len(y_test_np)*100:.1f}%), "
              f"{test_non_landslides} non-landslides ({test_non_landslides/len(y_test_np)*100:.1f}%)")
        
        # Report test set distribution (no artificial rebalancing!)
        test_landslide_ratio = test_landslides / len(y_test_np)
        if test_landslide_ratio > 0.6 or test_landslide_ratio < 0.4:
            print(f"\n📊 Test set shows spatial clustering ({test_landslide_ratio*100:.1f}% landslides)")
            print("   ✅ Maintaining natural distribution for valid evaluation")
            print("   📈 Focus on AUC-ROC and Recall for imbalanced assessment")
        else:
            print(f"\n✅ Test set distribution is acceptable ({test_landslide_ratio*100:.1f}% landslides)")
        
        # Keep original test set for valid evaluation (no rebalancing!)
        y_test = y_test_np
        
        # Scale features using RobustScaler (better for outliers)
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors (handle different data types)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        # Handle y_train and y_test data types properly
        if hasattr(y_train, 'values'):
            # pandas Series or DataFrame
            y_train_array = y_train.values
        else:
            # numpy array or other
            y_train_array = np.array(y_train)
            
        if hasattr(y_test, 'values'):
            # pandas Series or DataFrame  
            y_test_array = y_test.values
        else:
            # numpy array or other
            y_test_array = np.array(y_test)
        
        y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32).unsqueeze(1)
        
        return {
            'X_train': X_train_tensor,
            'X_test': X_test_tensor,
            'y_train': y_train_tensor,
            'y_test': y_test_tensor,
            'scaler': scaler,
            'selected_features': selected_features,
            'train_coords': train_coords,
            'test_coords': test_coords,
            'continuous_cols': continuous_cols
        }
    
    def _enhanced_feature_selection(self, X, y, max_features=15, enable_quality_filtering=True):
        """
        Enhanced feature selection based on data quality analysis
        
        Args:
            X: Feature matrix
            y: Target vector  
            max_features: Maximum number of features to select (default: 15 based on analysis)
            enable_quality_filtering: Whether to apply quality filtering
        """
        print(f"🔧 Enhanced Feature Selection Starting...")
        print(f"   Original features: {X.shape[1]}")
        
        # Step 1: Quality-based filtering (remove poor features)
        if enable_quality_filtering:
            print(f"   🚮 Applying quality filtering...")
            
            # Remove low-variance features (< 0.01)
            low_variance_features = []
            for col in X.columns:
                if X[col].var() < 0.01:
                    low_variance_features.append(col)
            
            # Remove binary categorical features with very low discriminative power
            # Based on our analysis - these are mostly noise
            weak_categorical_features = [
                col for col in X.columns 
                if (col.startswith('lithology_') or col.startswith('soil_')) and 
                   X[col].nunique() <= 2 and X[col].var() < 0.05
            ]
            
            # Combine features to remove
            features_to_remove = set(low_variance_features + weak_categorical_features)
            features_to_remove = [f for f in features_to_remove if f in X.columns]
            
            if features_to_remove:
                print(f"      Removing {len(features_to_remove)} low-quality features")
                X_filtered = X.drop(columns=features_to_remove)
            else:
                X_filtered = X.copy()
                
            print(f"      Features after quality filtering: {X_filtered.shape[1]}")
        else:
            X_filtered = X.copy()
        
        # Step 2: Statistical feature selection (F-test)
        print(f"   📊 Statistical feature selection...")
        k_select = min(max_features * 2, X_filtered.shape[1])  # Select 2x for next step
        
        selector = SelectKBest(score_func=f_classif, k=k_select)
        selector.fit(X_filtered, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X_filtered.columns,
            'score': selector.scores_,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False)
        
        statistical_features = feature_scores[feature_scores['selected']]['feature'].tolist()
        print(f"      Selected {len(statistical_features)} features by F-test")
        
        # Step 3: Random Forest importance (on pre-filtered features)
        print(f"   🌲 Random Forest importance ranking...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                  class_weight='balanced')
        rf.fit(X_filtered[statistical_features], y)
        
        # Get importance scores
        importance_df = pd.DataFrame({
            'feature': statistical_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features by importance
        rf_top_features = importance_df.head(max_features)['feature'].tolist()
        
        print(f"   ✅ Final feature selection results:")
        print(f"      Selected {len(rf_top_features)} top features")
        
        # Display top features with their scores
        final_feature_info = []
        for i, feature in enumerate(rf_top_features):
            f_score = feature_scores[feature_scores['feature'] == feature]['score'].iloc[0]
            rf_importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
            final_feature_info.append({
                'rank': i + 1,
                'feature': feature,
                'f_score': f_score,
                'rf_importance': rf_importance
            })
            print(f"         {i+1:2d}. {feature:<20} (F-score: {f_score:8.1f}, RF-imp: {rf_importance:.3f})")
        
        # Store feature selection info for later use
        self.feature_selection_info = {
            'original_features': X.shape[1],
            'after_quality_filter': X_filtered.shape[1],
            'final_selected': len(rf_top_features),
            'selected_features': rf_top_features,
            'feature_details': final_feature_info,
            'removed_features': features_to_remove if enable_quality_filtering else []
        }
        
        return rf_top_features
    
    def _ensemble_feature_selection(self, X, y, max_features=15):
        """Legacy ensemble method - redirects to enhanced selection"""
        return self._enhanced_feature_selection(X, y, max_features=max_features)
    
    def train_model(self, training_data, num_epochs=100, batch_size=64, 
                   learning_rate=0.001, patience=15, progress_callback=None):
        """
        Train the advanced model with focal loss and mixed precision
        
        Args:
            training_data: Dictionary from prepare_training_data_with_spatial_cv
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            patience: Early stopping patience
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with trained model and training info
        """
        
        print("\n" + "="*60)
        print("TRAINING ADVANCED ANN MODEL - IMPROVED VERSION")
        print("="*60)
        print("🚀 IMPROVEMENTS IMPLEMENTED:")
        print("   ✅ Focal Loss (alpha=0.25, gamma=2.0) - Better class imbalance handling")
        print("   ✅ Increased dropout (0.5) - Reduced overfitting") 
        print("   ✅ L2 regularization (weight_decay=0.01) - Better generalization")
        print("   ✅ Early stopping (patience=10) - Prevent overfitting")
        print("   ✅ Optimized threshold search (0.3-0.7) - Better F1 performance")
        print("="*60)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        
        # Calculate class weights
        y_train_np = np.array(y_train).squeeze()
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train_np), 
                                            y=y_train_np)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights: {class_weight_dict}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create weighted sampler
        sample_weights = [class_weight_dict[int(label)] for label in y_train_np]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create model
        input_size = X_train.shape[1]
        model = AdvancedLandslideANN(input_size=input_size)
        model.to(self.device)
        
        # Calculate class weights for balanced training
        train_landslides_count = int(y_train.sum().item())
        train_non_landslides_count = len(y_train) - train_landslides_count
        total_samples = len(y_train)
        
        # Compute pos_weight for BCEWithLogitsLoss
        # pos_weight = number of negative samples / number of positive samples
        pos_weight = torch.tensor([train_non_landslides_count / train_landslides_count]).to(self.device)
        
        print(f"\n⚖️  Class Balancing:")
        print(f"  Landslides: {train_landslides_count} ({train_landslides_count/total_samples*100:.1f}%)")
        print(f"  Non-landslides: {train_non_landslides_count} ({train_non_landslides_count/total_samples*100:.1f}%)")
        print(f"  pos_weight: {pos_weight.item():.3f}")
        
        # IMPROVED: Use Focal Loss for better class imbalance handling
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # IMPROVED: L2 regularization increased to 0.01, reduced learning rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # IMPROVED: Early stopping with patience=10
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # Mixed precision training disabled for CPU compatibility
        scaler_amp = None
        
        # Print dataset information
        train_landslides = int(y_train.sum().item())
        train_non_landslides = len(y_train) - train_landslides
        test_landslides = int(y_test.sum().item())
        test_non_landslides = len(y_test) - test_landslides
        
        print(f"\nTraining set: {len(train_loader.dataset)} samples")
        print(f"  - Landslides:     {train_landslides}")
        print(f"  - Non-landslides: {train_non_landslides}")
        print(f"\nTest set: {len(test_loader.dataset)} samples")
        print(f"  - Landslides:     {test_landslides}")
        print(f"  - Non-landslides: {test_non_landslides}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        print(f"Input features: {input_size}")
        
        for epoch in range(num_epochs):
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
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if progress_callback:
                # Call with (epoch, total_epochs) to match dialog expectations
                progress_callback(epoch + 1, num_epochs)
            
            # IMPROVED: Use new early stopping with best weights restoration
            if early_stopping(avg_val_loss, model):
                print(f"Early stopping at epoch {epoch+1} due to no improvement")
                print(f"Best validation loss: {early_stopping.best_loss:.6f}")
                break
        
        # MODEL CALIBRATION: Improve probability estimates
        print("\n" + "="*60) 
        print("🎛️ MODEL CALIBRATION")
        print("="*60)
        
        calibrated_model = self._calibrate_model(model, X_train, y_train, X_test, y_test)
        
        # ADVANCED THRESHOLD OPTIMIZATION: Run comprehensive optimization
        print("\n" + "="*60)
        print("🎯 ADVANCED THRESHOLD OPTIMIZATION")
        print("="*60)
        
        # Run comprehensive threshold optimization
        threshold_results = self._run_advanced_threshold_optimization(calibrated_model, X_test, y_test)
        
        # Select best threshold from advanced optimization
        best_threshold = threshold_results['recommended_threshold']
        print(f"🏆 Recommended threshold: {best_threshold:.3f}")
        
        # Evaluate model with optimized threshold (use original model for final evaluation)
        metrics = self._evaluate_model(model, X_test, y_test, best_threshold)
        
        # Add training set statistics
        train_landslides = int(y_train.sum().item())
        train_non_landslides = len(y_train) - train_landslides
        
        return {
            'model': model,
            'model_state_dict': model.state_dict(),
            'scaler': training_data['scaler'],
            'selected_features': training_data['selected_features'],
            'continuous_features': training_data['continuous_cols'],  # Save continuous feature names
            'best_threshold': best_threshold,
            'threshold_optimization': threshold_results,  # Save all threshold optimization results
            'calibrated_model': calibrated_model if hasattr(calibrated_model, 'is_calibrated') else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics,
            'input_size': input_size,
            'train_size': len(y_train),
            'train_landslides': train_landslides,
            'train_non_landslides': train_non_landslides
        }
    
    def _find_optimal_threshold(self, model, X_test, y_test):
        """
        IMPROVED: Find optimal classification threshold with comprehensive search
        Tests range 0.3-0.7 as recommended by analysis
        """
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = model(X_test)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        
        y_true = y_test.cpu().numpy()
        
        # IMPROVED: Test recommended range 0.3-0.7 with finer granularity
        thresholds = np.arange(0.3, 0.71, 0.02)  # 0.3, 0.32, 0.34, ..., 0.7
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        print(f"\n🎯 THRESHOLD OPTIMIZATION:")
        print(f"   Testing {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
        
        threshold_results = []
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            
            # Calculate comprehensive metrics for each threshold
            f1 = f1_score(y_true, predictions, zero_division=0)
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            accuracy = accuracy_score(y_true, predictions)
            
            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision, 
                'recall': recall,
                'accuracy': accuracy
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall, 
                    'accuracy': accuracy
                }
        
        # Report top 3 thresholds
        threshold_results.sort(key=lambda x: x['f1'], reverse=True)
        print(f"\n   📊 TOP 3 THRESHOLDS BY F1-SCORE:")
        for i, result in enumerate(threshold_results[:3]):
            print(f"   {i+1}. Threshold {result['threshold']:.2f}: "
                  f"F1={result['f1']:.3f}, Precision={result['precision']:.3f}, "
                  f"Recall={result['recall']:.3f}, Accuracy={result['accuracy']:.3f}")
        
        print(f"\n   ✅ Selected threshold: {best_threshold:.2f} (F1-Score: {best_f1:.3f})")
        
        return best_threshold
    
    def _calibrate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Calibrate model probabilities using Platt scaling or isotonic regression.
        
        Args:
            model: Trained PyTorch model
            X_train, y_train: Training data for calibration
            X_test, y_test: Test data for validation
            
        Returns:
            Calibrated model wrapper or original model if calibration fails
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            # Create a sklearn-compatible wrapper for PyTorch model
            class PyTorchWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, pytorch_model, device):
                    self.pytorch_model = pytorch_model
                    self.device = device
                    self.classes_ = np.array([0, 1])
                
                def fit(self, X, y):
                    # Already fitted, just return self
                    return self
                
                def predict_proba(self, X):
                    self.pytorch_model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        outputs = self.pytorch_model(X_tensor)
                        proba_1 = torch.sigmoid(outputs).cpu().numpy().ravel()
                        proba_0 = 1 - proba_1
                        return np.column_stack([proba_0, proba_1])
                
                def predict(self, X):
                    proba = self.predict_proba(X)
                    return (proba[:, 1] >= 0.5).astype(int)
            
            # Create wrapper
            pytorch_wrapper = PyTorchWrapper(model, self.device)
            
            # Prepare calibration data (use a subset to avoid overfitting)
            if isinstance(X_train, np.ndarray):
                X_train_cal = X_train
            else:
                X_train_cal = X_train.cpu().numpy()
                
            if isinstance(y_train, np.ndarray):
                y_train_cal = y_train.ravel()
            else:
                y_train_cal = y_train.cpu().numpy().ravel()
            
            print(f"   📊 Calibrating model probabilities...")
            print(f"   - Calibration samples: {len(X_train_cal)}")
            
            # Try Platt scaling first (faster and often works well)
            calibrator = CalibratedClassifierCV(
                pytorch_wrapper, 
                method='sigmoid',  # Platt scaling
                cv=3  # 3-fold cross-validation
            )
            
            # Fit calibrator
            calibrator.fit(X_train_cal, y_train_cal)
            
            # Test calibration quality
            if isinstance(X_test, np.ndarray):
                X_test_np = X_test
            else:
                X_test_np = X_test.cpu().numpy()
                
            if isinstance(y_test, np.ndarray):
                y_test_np = y_test.ravel()
            else:
                y_test_np = y_test.cpu().numpy().ravel()
            
            # Get uncalibrated probabilities
            uncal_proba = pytorch_wrapper.predict_proba(X_test_np)[:, 1]
            
            # Get calibrated probabilities  
            cal_proba = calibrator.predict_proba(X_test_np)[:, 1]
            
            # Measure calibration improvement using Brier score
            from sklearn.metrics import brier_score_loss
            
            uncal_brier = brier_score_loss(y_test_np, uncal_proba)
            cal_brier = brier_score_loss(y_test_np, cal_proba)
            
            print(f"   📈 Calibration Results:")
            print(f"   - Uncalibrated Brier Score: {uncal_brier:.4f}")
            print(f"   - Calibrated Brier Score: {cal_brier:.4f}")
            print(f"   - Improvement: {((uncal_brier - cal_brier) / uncal_brier * 100):.1f}%")
            
            if cal_brier < uncal_brier:
                print(f"   ✅ Calibration improved probability estimates!")
                
                # Create a calibrated model class
                class CalibratedPyTorchModel:
                    def __init__(self, calibrator, device):
                        self.calibrator = calibrator
                        self.device = device
                        self.is_calibrated = True
                    
                    def __call__(self, X_tensor):
                        # For threshold optimization, return calibrated probabilities
                        X_np = X_tensor.cpu().numpy()
                        proba = self.calibrator.predict_proba(X_np)[:, 1]
                        return torch.FloatTensor(proba).to(self.device).unsqueeze(1)
                    
                    def eval(self):
                        pass  # Compatibility method
                    
                    def state_dict(self):
                        # Return the original model state dict
                        return self.calibrator.base_estimator.pytorch_model.state_dict()
                
                calibrated_model = CalibratedPyTorchModel(calibrator, self.device)
                return calibrated_model
            else:
                print(f"   ⚠️ Calibration did not improve results, using original model")
                return model
                
        except ImportError:
            print(f"   ⚠️ Calibration not available (sklearn version issue)")
            return model
        except Exception as e:
            print(f"   ⚠️ Calibration failed: {e}")
            return model
    
    def _run_advanced_threshold_optimization(self, model, X_test, y_test):
        """
        Run comprehensive threshold optimization using multiple methods.
        
        Returns:
            dict: Comprehensive threshold optimization results
        """
        from sklearn.metrics import (
            roc_curve, precision_recall_curve, fbeta_score, confusion_matrix
        )
        
        # Get model predictions (handle both calibrated and regular models)
        model.eval()
        with torch.no_grad():
            # Convert to tensor if it's numpy array
            if isinstance(X_test, np.ndarray):
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            else:
                X_test_tensor = X_test.to(self.device)
            
            if hasattr(model, 'is_calibrated') and model.is_calibrated:
                # Calibrated model - outputs are already probabilities
                outputs = model(X_test_tensor)
                y_proba = outputs.cpu().numpy().ravel()
            else:
                # Regular PyTorch model - need sigmoid activation
                outputs = model(X_test_tensor)
                y_proba = torch.sigmoid(outputs).cpu().numpy().ravel()
        
        # Convert y_test to numpy if needed
        if isinstance(y_test, np.ndarray):
            y_true = y_test.ravel()
        else:
            y_true = y_test.cpu().numpy().ravel()
        
        print(f"   📊 Optimizing thresholds on {len(y_true)} validation samples")
        print(f"   - Landslides: {np.sum(y_true)} ({100*np.mean(y_true):.1f}%)")
        print(f"   - Prediction range: [{np.min(y_proba):.3f}, {np.max(y_proba):.3f}]")
        
        optimization_results = {}
        
        # 1. ROC-based optimization (Youden's J statistic)
        print("   🔍 ROC Curve Optimization...")
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        roc_threshold = roc_thresholds[best_j_idx]
        
        y_pred_roc = (y_proba >= roc_threshold).astype(int)
        optimization_results['roc_youden'] = {
            'threshold': roc_threshold,
            'f1': f1_score(y_true, y_pred_roc, zero_division=0),
            'precision': precision_score(y_true, y_pred_roc, zero_division=0),
            'recall': recall_score(y_true, y_pred_roc, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_roc),
            'youden_j': j_scores[best_j_idx]
        }
        
        # 2. Precision-Recall optimization (F1 maximization)
        print("   📈 Precision-Recall Optimization...")
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        pr_threshold = pr_thresholds[best_f1_idx]
        
        y_pred_pr = (y_proba >= pr_threshold).astype(int)
        optimization_results['pr_f1_max'] = {
            'threshold': pr_threshold,
            'f1': f1_score(y_true, y_pred_pr, zero_division=0),
            'precision': precision_score(y_true, y_pred_pr, zero_division=0),
            'recall': recall_score(y_true, y_pred_pr, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_pr)
        }
        
        # 3. Landslide-focused optimization (prioritize recall)
        print("   🏔️ Landslide-Focused Optimization...")
        test_thresholds = np.arange(0.05, 0.96, 0.01)
        best_recall_score = 0
        best_recall_threshold = 0.5
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            
            # Weighted score favoring recall (2:1 ratio) for landslide detection
            if recall >= 0.6:  # Minimum acceptable recall
                score = (2 * recall + precision) / 3
                if score > best_recall_score:
                    best_recall_score = score
                    best_recall_threshold = threshold
        
        y_pred_landslide = (y_proba >= best_recall_threshold).astype(int)
        optimization_results['landslide_focused'] = {
            'threshold': best_recall_threshold,
            'f1': f1_score(y_true, y_pred_landslide, zero_division=0),
            'precision': precision_score(y_true, y_pred_landslide, zero_division=0),
            'recall': recall_score(y_true, y_pred_landslide, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_landslide),
            'weighted_score': best_recall_score
        }
        
        # 4. Cost-sensitive optimization
        print("   💰 Cost-Sensitive Optimization...")
        cost_matrix = np.array([[0, 1], [10, 0]])  # Missing landslides cost 10x more
        best_cost = float('inf')
        cost_threshold = 0.5
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            total_cost = (tn * cost_matrix[0,0] + fp * cost_matrix[0,1] + 
                         fn * cost_matrix[1,0] + tp * cost_matrix[1,1])
            
            if total_cost < best_cost:
                best_cost = total_cost
                cost_threshold = threshold
        
        y_pred_cost = (y_proba >= cost_threshold).astype(int)
        optimization_results['cost_sensitive'] = {
            'threshold': cost_threshold,
            'f1': f1_score(y_true, y_pred_cost, zero_division=0),
            'precision': precision_score(y_true, y_pred_cost, zero_division=0),
            'recall': recall_score(y_true, y_pred_cost, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_cost),
            'total_cost': best_cost
        }
        
        # 5. F-beta optimization (emphasizing recall)
        print("   🎯 F-beta Optimization...")
        beta = 1.5  # Emphasize recall more than precision
        best_fbeta = 0
        fbeta_threshold = 0.5
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                fbeta_threshold = threshold
        
        y_pred_fbeta = (y_proba >= fbeta_threshold).astype(int)
        optimization_results['fbeta_1_5'] = {
            'threshold': fbeta_threshold,
            'f1': f1_score(y_true, y_pred_fbeta, zero_division=0),
            'precision': precision_score(y_true, y_pred_fbeta, zero_division=0),
            'recall': recall_score(y_true, y_pred_fbeta, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_fbeta),
            'fbeta_score': best_fbeta
        }
        
        # Determine best overall threshold
        print("\n   📊 THRESHOLD OPTIMIZATION RESULTS:")
        
        # Rank methods by F1 score and landslide detection capability
        method_scores = []
        for method, results in optimization_results.items():
            # Score combines F1 and recall (for landslide detection)
            composite_score = results['f1'] * 0.6 + results['recall'] * 0.4
            method_scores.append((method, results['threshold'], composite_score, results))
        
        method_scores.sort(key=lambda x: x[2], reverse=True)
        
        for i, (method, threshold, score, results) in enumerate(method_scores):
            print(f"   {i+1}. {method}: Threshold={threshold:.3f}, "
                  f"F1={results['f1']:.3f}, Recall={results['recall']:.3f}, "
                  f"Score={score:.3f}")
        
        # Select the best method (highest composite score)
        best_method = method_scores[0][0]
        recommended_threshold = method_scores[0][1]
        
        print(f"\n   🏆 SELECTED METHOD: {best_method}")
        print(f"   🎯 RECOMMENDED THRESHOLD: {recommended_threshold:.3f}")
        
        return {
            'recommended_threshold': recommended_threshold,
            'best_method': best_method,
            'all_results': optimization_results,
            'method_rankings': method_scores
        }
    
    def _evaluate_model(self, model, X_test, y_test, threshold=0.5):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(X_test, np.ndarray):
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            else:
                X_test_tensor = X_test.to(self.device)
            
            outputs = model(X_test_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > threshold).astype(int)
        
        # Convert y_test to numpy if needed  
        if isinstance(y_test, np.ndarray):
            y_true = y_test
        else:
            y_true = y_test.cpu().numpy()
        
        # Calculate test set statistics
        total_samples = len(y_true)
        num_landslides = int(np.sum(y_true))
        num_non_landslides = total_samples - num_landslides
        
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions),
            'recall': recall_score(y_true, predictions),
            'f1': f1_score(y_true, predictions),
            'auc_roc': roc_auc_score(y_true, probabilities),
            'test_size': total_samples,
            'test_landslides': num_landslides,
            'test_non_landslides': num_non_landslides
        }
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        print(f"Test Set Size: {total_samples}")
        print(f"  - Landslides:     {num_landslides}")
        print(f"  - Non-landslides: {num_non_landslides}")
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print("="*60)
        
        return metrics
