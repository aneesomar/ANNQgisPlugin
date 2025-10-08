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
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.6):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
        
        # Ensemble feature selection
        print("\nPerforming ensemble feature selection...")
        selected_features = self._ensemble_feature_selection(X, y, max_features=60)
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
        
        # If test set is imbalanced (> 60% either class), resample it
        test_landslide_ratio = test_landslides / len(y_test_np)
        if test_landslide_ratio > 0.6 or test_landslide_ratio < 0.4:
            print(f"\n⚠️  Test set imbalanced ({test_landslide_ratio*100:.1f}% landslides)!")
            print("   Resampling test set to match training distribution...")
            
            # Calculate target ratio from training set
            target_ratio = train_landslides / len(y_train_np)
            target_test_landslides = int(len(y_test_np) * target_ratio)
            target_test_non_landslides = len(y_test_np) - target_test_landslides
            
            # Get indices of landslides and non-landslides in test set
            test_landslide_indices = np.where(y_test_np == 1)[0]
            test_non_landslide_indices = np.where(y_test_np == 0)[0]
            
            # Resample to target distribution
            np.random.seed(42)
            if len(test_landslide_indices) > target_test_landslides:
                # Downsample landslides
                sampled_landslide_indices = np.random.choice(test_landslide_indices, 
                                                              target_test_landslides, 
                                                              replace=False)
            else:
                # Upsample landslides
                sampled_landslide_indices = np.random.choice(test_landslide_indices, 
                                                              target_test_landslides, 
                                                              replace=True)
            
            if len(test_non_landslide_indices) > target_test_non_landslides:
                # Downsample non-landslides
                sampled_non_landslide_indices = np.random.choice(test_non_landslide_indices, 
                                                                  target_test_non_landslides, 
                                                                  replace=False)
            else:
                # Upsample non-landslides
                sampled_non_landslide_indices = np.random.choice(test_non_landslide_indices, 
                                                                  target_test_non_landslides, 
                                                                  replace=True)
            
            # Combine and shuffle
            resampled_indices = np.concatenate([sampled_landslide_indices, sampled_non_landslide_indices])
            np.random.shuffle(resampled_indices)
            
            # Apply resampling (handle different data types)
            if isinstance(X_test, torch.Tensor):
                X_test = X_test[resampled_indices]
                y_test = y_test[resampled_indices]
            elif isinstance(X_test, pd.DataFrame):
                # For pandas DataFrame, use .iloc for positional indexing
                X_test = X_test.iloc[resampled_indices].reset_index(drop=True)
                y_test = y_test.iloc[resampled_indices].reset_index(drop=True) if isinstance(y_test, pd.Series) else y_test[resampled_indices]
            else:
                # For numpy arrays
                X_test = X_test[resampled_indices]
                y_test = y_test[resampled_indices]
            
            if test_coords is not None:
                if isinstance(test_coords, pd.DataFrame):
                    test_coords = test_coords.iloc[resampled_indices].reset_index(drop=True)
                else:
                    test_coords = test_coords[resampled_indices]
            
            # Verify new balance (handle different data types)
            if isinstance(y_test, torch.Tensor):
                y_test_np = y_test.numpy()
            elif isinstance(y_test, pd.Series):
                y_test_np = y_test.values
            else:
                y_test_np = np.array(y_test).flatten()
            
            new_test_landslides = int(np.sum(y_test_np))
            new_test_non_landslides = len(y_test_np) - new_test_landslides
            
            print(f"\n✅ Test set rebalanced:")
            print(f"   {new_test_landslides} landslides ({new_test_landslides/len(y_test_np)*100:.1f}%), "
                  f"{new_test_non_landslides} non-landslides ({new_test_non_landslides/len(y_test_np)*100:.1f}%)")
        else:
            print(f"\n✅ Test set balance looks good ({test_landslide_ratio*100:.1f}% landslides)")
        
        # Scale features using RobustScaler (better for outliers)
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        
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
    
    def _ensemble_feature_selection(self, X, y, max_features=60):
        """Ensemble feature selection with voting"""
        print(f"Number of features before selection: {X.shape[1]}")
        
        # Method 1: Statistical (F-test)
        selector = SelectKBest(score_func=f_classif, k=min(max_features, X.shape[1]))
        selector.fit(X, y)
        kbest_features = X.columns[selector.get_support()].tolist()
        
        # Method 2: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        rf_features = [X.columns[i] for i in indices[:max_features]]
        
        # Method 3: RFE
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                  n_features_to_select=max_features)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_].tolist()
        
        # Voting system
        all_features = set(kbest_features + rf_features + rfe_features)
        feature_votes = {}
        for feature in all_features:
            votes = 0
            if feature in kbest_features: votes += 1
            if feature in rf_features: votes += 1
            if feature in rfe_features: votes += 1
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes
        final_features = [f for f, votes in feature_votes.items() if votes >= 2]
        print(f"Features selected by ensemble method: {len(final_features)}")
        
        return final_features
    
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
        print("TRAINING ADVANCED ANN MODEL")
        print("="*60)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        
        # Calculate class weights
        y_train_np = y_train.squeeze().numpy()
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train_np), 
                                            y=y_train_np)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights: {class_weight_dict}")
        
        # Create weighted sampler
        sample_weights = [class_weight_dict[int(label)] for label in y_train_np]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
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
        
        # Loss and optimizer - UPDATED with class balancing
        # BCEWithLogitsLoss with pos_weight balances the dataset
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Reduced learning rate for better convergence
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Mixed precision training
        scaler_amp = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
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
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Find optimal threshold
        print("\nFinding optimal threshold...")
        best_threshold = self._find_optimal_threshold(model, X_test, y_test)
        print(f"Optimal threshold: {best_threshold:.3f}")
        
        # Evaluate model
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
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics,
            'input_size': input_size,
            'train_size': len(y_train),
            'train_landslides': train_landslides,
            'train_non_landslides': train_non_landslides
        }
    
    def _find_optimal_threshold(self, model, X_test, y_test):
        """Find optimal classification threshold"""
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = model(X_test)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        
        y_true = y_test.cpu().numpy()
        
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            f1 = f1_score(y_true, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_model(self, model, X_test, y_test, threshold=0.5):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            outputs = model(X_test)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > threshold).astype(int)
        
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
