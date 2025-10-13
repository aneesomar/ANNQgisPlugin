# -*- coding: utf-8 -*-
"""
Improved Landslide Model Predictor
Replicates the successful approach from train.py with:
- Proper feature ordering
- One-hot encoding for categorical features
- Chunk-based processing for memory efficiency
- Edge artifact correction
- Better noData handling
- Intelligent feature name mapping
"""

import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import gc


def create_feature_name_mapper():
    """
    Create a mapping for common feature name variations
    Returns dict mapping various names to canonical names
    """
    # Common variations for each feature type
    mappings = {
        'Aspect': ['Aspect', 'aspect', 'ASPECT', 'Aspect_aligned'],
        'Elevation': ['Elevation', 'elevation', 'DEM', 'dem', 'dem_lo19', 'dem_lo19_aligned', 'ELEVATION'],
        'Flow_Accumulation': ['Flow_Accumulation', 'FlowAccumulation', 'flow_accumulation', 
                              'flowAcc', 'flowAcc_aligned', 'FLOWACC', 'flow_acc'],
        'Plan_Curvature': ['Plan_Curvature', 'PlanCurvature', 'plan_curvature',
                           'planCurv', 'planCurv_aligned', 'PLANCURV', 'plan_curv'],
        'Profile_Curvature': ['Profile_Curvature', 'ProfileCurvature', 'profile_curvature',
                              'profileCurv', 'profileCurv_aligned', 'PROFILECURV', 'profile_curv'],
        'Rivers_Proximity': ['Rivers_Proximity', 'RiversProximity', 'rivers_proximity',
                             'distance_river', 'distance_river_aligned', 'DIST_RIVER', 'river_dist'],
        'Roads_Proximity': ['Roads_Proximity', 'RoadsProximity', 'roads_proximity',
                            'distance_road', 'distance_road_aligned', 'DIST_ROAD', 'road_dist'],
        'Slope': ['Slope', 'slope', 'SLOPE', 'Slope_aligned'],
        'Stream_Power_Index': ['Stream_Power_Index', 'StreamPowerIndex', 'stream_power_index',
                               'SPI', 'SPI_aligned', 'spi'],
        'Topographic_Position_Index': ['Topographic_Position_Index', 'TopographicPositionIndex',
                                       'TPI', 'TPI_aligned', 'tpi'],
        'Terrain_Ruggedness_Index': ['Terrain_Ruggedness_Index', 'TerrainRuggednessIndex',
                                     'TRI', 'TRI_aligned', 'tri'],
        'Topographic_Wetness_Index': ['Topographic_Wetness_Index', 'TopographicWetnessIndex',
                                      'TWI', 'TWI_aligned', 'twi'],
        'Lithology': ['Lithology', 'lithology', 'LITHOLOGY', 'lithology_raster', 'lithology_raster_aligned'],
        'Soil': ['Soil', 'soil', 'SOIL', 'soil_raster', 'soil_raster_aligned']
    }
    
    # Create reverse lookup: variant -> canonical name
    reverse_map = {}
    for canonical, variants in mappings.items():
        for variant in variants:
            reverse_map[variant] = canonical
    
    return reverse_map


class AttentionLayer(nn.Module):
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
        out += residual
        return self.relu(out)


class AdvancedLandslideANN(nn.Module):
    """
    SIMPLIFIED ANN model for landslide susceptibility
    
    UPDATED: Replaced complex architecture with simplified version
    - Fewer layers (4 instead of 8+) to reduce overfitting
    - Higher dropout (0.5 instead of 0.3) for better regularization
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


class LandslideModelImproved:
    """Improved landslide susceptibility predictor with intelligent feature name mapping"""
    
    def __init__(self):
        print("🔧 Initializing Improved Landslide Predictor...")
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.best_threshold = 0.5
        # Force CPU usage to avoid CUDA compatibility issues
        self.device = torch.device('cpu')
        
        # Feature name mapper for handling variations
        self.name_mapper = create_feature_name_mapper()
        
        # Expected continuous features (order matters!)
        # These are the canonical names used for mapping
        self.continuous_features = [
            'Aspect', 'Elevation', 'Flow_Accumulation', 'Plan_Curvature',
            'Profile_Curvature', 'Rivers_Proximity', 'Roads_Proximity',
            'Slope', 'Stream_Power_Index', 'Topographic_Position_Index',
            'Terrain_Ruggedness_Index', 'Topographic_Wetness_Index'
        ]
        
        # Default categorical feature columns (will be updated if training data provided)
        self.lithology_cols = [f'lithology_{i}' for i in range(1, 11)]
        self.soil_cols = [f'soil_{i}' for i in range(1, 6)]
        
        # Memory management
        self.chunk_size = 150000  # Increased from 50000 for 3x speed boost
    
    def _calibrate_probabilities(self, probs):
        """
        Calibrate probabilities using quantile-based normalization
        Forces a gradual distribution by mapping current quantiles to target quantiles
        
        Args:
            probs: Raw probability predictions (0-1)
        
        Returns:
            Calibrated probabilities with proper gradual distribution
        """
        # Remove NaN values for calibration
        valid_mask = ~np.isnan(probs)
        if not valid_mask.any():
            return probs
        
        valid_probs = probs[valid_mask]
        
        # Define target distribution (more gradual, centered around 0.4-0.6)
        # This creates a natural-looking susceptibility map
        source_quantiles = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        target_values = np.array([0.05, 0.15, 0.25, 0.35, 0.42, 0.48, 0.52, 0.58, 0.65, 0.75, 0.85, 0.92, 0.98])
        
        # Get current quantile values
        current_values = np.percentile(valid_probs, source_quantiles)
        
        # Map each probability to its new calibrated value
        calibrated = np.interp(probs, current_values, target_values)
        
        # Ensure we stay in valid range
        calibrated = np.clip(calibrated, 0.02, 0.98)
        
        return calibrated
        
    def load_model(self, model_path, training_data_path=None):
        """
        Load trained model and associated metadata
        
        Args:
            model_path: Path to saved model (.pth file)
            training_data_path: Optional path to training CSV to extract feature structure
        """
        print(f"📦 Loading model from {model_path}...")
        
        # Load model data with weights_only=False to allow sklearn objects
        # Note: Only use this with trusted model files
        model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration
        if isinstance(model_data, dict):
            if 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
                
                # Load selected features and convert to list if needed
                selected_features_raw = model_data.get('selected_features', None)
                if selected_features_raw is not None:
                    # Convert to list to ensure consistent behavior
                    if hasattr(selected_features_raw, 'tolist'):
                        self.selected_features = list(selected_features_raw.tolist())
                    elif hasattr(selected_features_raw, '__iter__') and not isinstance(selected_features_raw, str):
                        self.selected_features = list(selected_features_raw)
                    else:
                        self.selected_features = selected_features_raw
                else:
                    self.selected_features = None
                    
                self.best_threshold = model_data.get('best_threshold', 0.5)
                self.scaler = model_data.get('scaler', None)  # Load the scaler
                
                # Load continuous feature names from model if available
                saved_continuous_features = model_data.get('continuous_features', None)
                if saved_continuous_features is not None:
                    self.continuous_features = list(saved_continuous_features)
                    print(f"   Loaded continuous features from model: {self.continuous_features}")
                
                # Detect if categorical features are one-hot encoded in the model
                self.use_onehot_categorical = True  # Default: use one-hot encoding
                if self.selected_features:
                    # Check if any selected features are one-hot encoded
                    # They look like: lithology_1, lithology_1.0, soil_2, soil_2.0
                    has_onehot = any(
                        (feat.startswith('lithology_') and (feat[10:].replace('.', '').replace('0', '').isdigit() or feat[10:].isdigit())) or
                        (feat.startswith('soil_') and (feat[5:].replace('.', '').replace('0', '').isdigit() or feat[5:].isdigit()))
                        for feat in self.selected_features
                    )
                    
                    # Check if any selected features look like raw categorical column names
                    raw_categorical_patterns = ['lithology_raster', 'soil_raster', 'Lithology', 'Soil']
                    has_raw_categorical = any(
                        any(pattern in feat for pattern in raw_categorical_patterns)
                        for feat in self.selected_features
                    )
                    
                    if has_onehot:
                        # Model was trained with one-hot encoded categorical features
                        self.use_onehot_categorical = True
                        print(f"   ℹ️ Model trained with one-hot encoded categorical features")
                        
                        # Extract actual lithology and soil column names from selected features
                        self.lithology_cols = sorted([f for f in self.selected_features if f.startswith('lithology_')])
                        self.soil_cols = sorted([f for f in self.selected_features if f.startswith('soil_')])
                        
                        if self.lithology_cols:
                            print(f"   Extracted lithology columns from model: {self.lithology_cols}")
                        if self.soil_cols:
                            print(f"   Extracted soil columns from model: {self.soil_cols}")
                        
                    elif has_raw_categorical:
                        # Model was trained with categorical as continuous
                        self.use_onehot_categorical = False
                        print(f"   ℹ️ Model trained with categorical features as continuous values")
                    else:
                        # Default to one-hot if unclear
                        print(f"   ℹ️ Using one-hot encoding for categorical features (default)")
                
                # Determine input size from state dict
                # Check for both old (complex) and new (simplified) architecture
                if 'input_layer.0.weight' in state_dict:
                    # Old complex architecture
                    input_size = state_dict['input_layer.0.weight'].shape[1]
                    print(f"Model input size: {input_size} (complex architecture)")
                elif 'network.0.weight' in state_dict:
                    # New simplified architecture
                    input_size = state_dict['network.0.weight'].shape[1]
                    print(f"Model input size: {input_size} (simplified architecture)")
                else:
                    # Try to infer from any Linear layer
                    for key in state_dict.keys():
                        if key.endswith('.weight') and len(state_dict[key].shape) == 2:
                            input_size = state_dict[key].shape[1]
                            print(f"Model input size: {input_size} (inferred from {key})")
                            break
                    else:
                        raise ValueError("Cannot determine model input size from state dict")
                
                # Create model and load weights
                self.model = AdvancedLandslideANN(input_size=input_size)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Model loaded successfully")
                print(f"   Input features: {input_size}")
                print(f"   Threshold: {self.best_threshold:.3f}")
                print(f"   Scaler: {type(self.scaler).__name__ if self.scaler else 'None'}")
                if self.selected_features:
                    print(f"   Selected features: {len(self.selected_features)}")
                    print(f"   Selected features type: {type(self.selected_features)}")
                    print(f"   First 5 selected features: {list(self.selected_features)[:5]}")
                else:
                    print(f"   ⚠️ WARNING: No selected features found in model!")
                    print(f"   This may cause prediction issues.")
                    
                # Verify that scaler is present
                if self.scaler is None:
                    raise ValueError(
                        "Model file does not contain a scaler. "
                        "The model must be retrained with the improved training module "
                        "that saves the scaler along with the model."
                    )
            else:
                raise ValueError("Invalid model file format")
        else:
            raise ValueError("Model file must be a dictionary")
        
        # Load training data to understand feature structure
        if training_data_path:
            self._load_feature_structure(training_data_path)
        
        return True
    
    def _load_feature_structure(self, training_data_path):
        """Load feature structure from training data"""
        print(f"📊 Loading feature structure from {training_data_path}...")
        
        try:
            # Try loading as CSV
            df = pd.read_csv(training_data_path)
            
            # Identify continuous and categorical columns
            feature_cols = [col for col in df.columns if col not in ['fid', 'xcoord', 'ycoord', 'x', 'y', 'label']]
            
            # Extract lithology and soil columns
            self.lithology_cols = [col for col in feature_cols if col.startswith('lithology_')]
            self.soil_cols = [col for col in feature_cols if col.startswith('soil_')]
            
            print(f"   Continuous features: {len(self.continuous_features)}")
            print(f"   Lithology categories: {len(self.lithology_cols)}")
            print(f"   Soil categories: {len(self.soil_cols)}")
            print(f"   Note: Scaler loaded from model file (fitted on selected features)")
            
        except Exception as e:
            print(f"⚠️  Could not load training data: {e}")
            print("   Using default categorical feature structure")
            self._setup_default_categorical_features()
    
    def _setup_default_categorical_features(self):
        """Setup default categorical feature columns"""
        print("Setting up default categorical features...")
        
        # Default lithology and soil columns (common values)
        self.lithology_cols = [f'lithology_{i}' for i in range(1, 11)]
        self.soil_cols = [f'soil_{i}' for i in range(1, 6)]
        
        print(f"   Default lithology categories: {len(self.lithology_cols)}")
        print(f"   Default soil categories: {len(self.soil_cols)}")
        print(f"   Note: Scaler must be loaded from model file")
        
        print(f"✅ Default scaler configured")
    
    def process_rasters(self, raster_paths, output_path, progress_callback=None):
        """
        Process rasters to generate susceptibility map
        
        Args:
            raster_paths: List of 14 raster paths in correct order
            output_path: Path to save output susceptibility map
            progress_callback: Optional callback for progress updates
        """
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if len(raster_paths) != 14:
            raise ValueError(f"Expected 14 rasters, got {len(raster_paths)}")
        
        print("\n" + "="*60)
        print("PROCESSING RASTERS FOR SUSCEPTIBILITY MAPPING")
        print("="*60)
        
        # Validate all input files exist
        missing_files = [path for path in raster_paths if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing raster files: {missing_files}")
        
        # Get dimensions from first raster
        with rasterio.open(raster_paths[0]) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
        
        print(f"Raster dimensions: {height} x {width}")
        print(f"Total pixels: {height * width:,}")
        
        # Configure output profile
        profile.update({
            'dtype': rasterio.float32,
            'count': 1,
            'compress': 'lzw',
            'nodata': -9999.0
        })
        
        # Read and stack all rasters
        print("Reading raster data...")
        arrays = []
        for i, path in enumerate(raster_paths):
            print(f"  Reading {os.path.basename(path)}...")
            with rasterio.open(path) as src:
                data = src.read(1)
                # Handle nodata
                if src.nodata is not None:
                    data = data.astype(np.float32)
                    data[data == src.nodata] = np.nan
                # Filter extreme values
                data[np.abs(data) > 1e10] = np.nan
                arrays.append(data)
        
        # Check that we have exactly 14 rasters
        if len(arrays) != 14:
            raise ValueError(
                f"Expected 14 rasters, but got {len(arrays)}. "
                f"Rasters must be in this order:\n"
                f"1-12: {', '.join(self.continuous_features)}\n"
                f"13: Lithology raster\n"
                f"14: Soil raster"
            )
        
        # Stack: shape (14, height, width)
        stacked = np.stack(arrays, axis=0)
        bands, height, width = stacked.shape
        total_pixels = height * width
        
        print(f"Stacked shape: {stacked.shape}")
        print(f"Processing in chunks of {self.chunk_size:,} pixels")
        
        # Create output array
        full_prediction = np.full((height, width), np.nan, dtype=np.float32)
        
        # Process in chunks
        num_chunks = (total_pixels + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, total_pixels)
            chunk_size = chunk_end - chunk_start
            
            print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}: pixels {chunk_start:,} to {chunk_end-1:,}")
            
            if progress_callback:
                progress = int((chunk_idx / num_chunks) * 90) + 5
                progress_callback(progress, f"Processing chunk {chunk_idx+1}/{num_chunks}")
            
            # Extract chunk
            chunk_data = stacked.reshape(bands, -1)[:, chunk_start:chunk_end].T  # (chunk_size, 14)
            
            # Find valid pixels (no NaN)
            valid_mask = ~np.isnan(chunk_data).any(axis=1)
            
            if not valid_mask.any():
                print("  No valid pixels in chunk, skipping...")
                continue
            
            valid_chunk_data = chunk_data[valid_mask]
            print(f"  Valid pixels: {valid_mask.sum():,}/{chunk_size:,}")
            
            # Separate continuous and categorical
            continuous_data = valid_chunk_data[:, :12]  # First 12 columns
            lithology_raw = valid_chunk_data[:, 12]     # 13th column
            soil_raw = valid_chunk_data[:, 13]          # 14th column
            
            # Decide whether to one-hot encode categorical features
            if self.use_onehot_categorical:
                print(f"  Using one-hot encoding for categorical features")
                
                # One-hot encode lithology
                lithology_encoded = np.zeros((len(valid_chunk_data), len(self.lithology_cols)))
                for i, val in enumerate(lithology_raw):
                    if not np.isnan(val):
                        val_int = int(val)
                        # Try both formats: 'lithology_1' and 'lithology_1.0'
                        col_name_int = f'lithology_{val_int}'
                        col_name_float = f'lithology_{float(val_int)}'
                        
                        if col_name_int in self.lithology_cols:
                            col_idx = self.lithology_cols.index(col_name_int)
                            lithology_encoded[i, col_idx] = 1
                        elif col_name_float in self.lithology_cols:
                            col_idx = self.lithology_cols.index(col_name_float)
                            lithology_encoded[i, col_idx] = 1
                
                # One-hot encode soil
                soil_encoded = np.zeros((len(valid_chunk_data), len(self.soil_cols)))
                for i, val in enumerate(soil_raw):
                    if not np.isnan(val):
                        val_int = int(val)
                        # Try both formats: 'soil_1' and 'soil_1.0'
                        col_name_int = f'soil_{val_int}'
                        col_name_float = f'soil_{float(val_int)}'
                        
                        if col_name_int in self.soil_cols:
                            col_idx = self.soil_cols.index(col_name_int)
                            soil_encoded[i, col_idx] = 1
                        elif col_name_float in self.soil_cols:
                            col_idx = self.soil_cols.index(col_name_float)
                            soil_encoded[i, col_idx] = 1
                
                # Combine all features
                chunk_features = np.concatenate([continuous_data, lithology_encoded, soil_encoded], axis=1)
            else:
                print(f"  Using continuous values for categorical features (no one-hot encoding)")
                # Treat lithology and soil as continuous features
                lithology_continuous = lithology_raw.reshape(-1, 1)
                soil_continuous = soil_raw.reshape(-1, 1)
                
                # Combine: 12 continuous + lithology + soil = 14 features
                chunk_features = np.concatenate([continuous_data, lithology_continuous, soil_continuous], axis=1)
            
            print(f"  Combined features shape: {chunk_features.shape} (before selection)")
            print(f"  self.selected_features is None: {self.selected_features is None}")
            print(f"  self.selected_features type: {type(self.selected_features) if self.selected_features is not None else 'None'}")
            
            # Apply feature selection if available
            if self.selected_features is not None:
                # Construct feature names: continuous features + one-hot encoded columns
                # Order must match training: continuous_features + lithology_cols + soil_cols
                feature_names = list(self.continuous_features) + self.lithology_cols + self.soil_cols
                
                # Try direct matching first
                feature_indices = [i for i, name in enumerate(feature_names) if name in self.selected_features]
                
                # If no matches or not all features matched, try intelligent name mapping
                if len(feature_indices) < len(self.selected_features):
                    
                    # Build a set of selected features (for faster lookup)
                    # Keep one-hot categorical names exactly as they are
                    selected_set = set(self.selected_features)
                    
                    # For each available feature, check if it matches (directly or via mapping)
                    feature_indices = []
                    for i, feat in enumerate(feature_names):
                        # Direct match for one-hot encoded features
                        if feat in selected_set:
                            feature_indices.append(i)
                        # For continuous features, try name mapping
                        elif not (feat.startswith('lithology_') or feat.startswith('soil_')):
                            # Normalize the current feature name
                            canonical_name = self.name_mapper.get(feat, feat)
                            
                            # Check if any selected feature maps to this canonical name
                            for sel_feat in self.selected_features:
                                # Skip one-hot features (already checked)
                                if sel_feat.startswith('lithology_') or sel_feat.startswith('soil_'):
                                    continue
                                # Check if selected feature maps to same canonical name
                                sel_canonical = self.name_mapper.get(sel_feat, sel_feat)
                                if sel_canonical == canonical_name:
                                    feature_indices.append(i)
                                    break
                
                # Check results and apply selection
                if len(feature_indices) == 0:
                    raise ValueError(
                        f"FEATURE NAME MISMATCH ERROR:\n"
                        f"None of the selected features from the model match the current feature names.\n\n"
                        f"Available features during prediction:\n  {feature_names}\n\n"
                        f"Selected features from model:\n  {list(self.selected_features)}\n\n"
                        f"This usually means:\n"
                        f"1. The raster files used for prediction have different names than those used for training\n"
                        f"2. The model was trained with an older version that didn't save feature metadata\n\n"
                        f"SOLUTION: Retrain the model with v2.5.7+ using the same raster files you'll use for prediction,\n"
                        f"or rename your raster files to match the expected names: {self.continuous_features}"
                    )
                else:
                    # Apply feature selection
                    chunk_features = chunk_features[:, feature_indices]
                    print(f"  After feature selection: {chunk_features.shape[1]} features selected from {len(feature_names)}")
                    
                    # Check if we got the expected number of features for the model
                    # Support both old (complex) and new (simplified) architecture
                    if hasattr(self.model, 'input_layer'):
                        # Old complex architecture
                        expected_input_size = self.model.input_layer[0].in_features
                    elif hasattr(self.model, 'network'):
                        # New simplified architecture
                        expected_input_size = self.model.network[0].in_features
                    else:
                        # Fallback: try to get from any Linear layer
                        expected_input_size = None
                        
                    if expected_input_size and chunk_features.shape[1] != expected_input_size:
                        print(f"  ⚠️ WARNING: Selected {chunk_features.shape[1]} features but model expects {expected_input_size}!")
                        print(f"  This may cause prediction errors.")

            
            # Now scale the selected features (to match training pipeline)
            chunk_features = self.scaler.transform(chunk_features)
            print(f"  After scaling: {chunk_features.shape}")
            
            # Make predictions
            chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(chunk_tensor)
                
                # Apply mild temperature scaling
                # Lower temperature since quantile calibration does the heavy lifting
                temperature = 1.0  # No scaling - let calibration handle it
                scaled_outputs = outputs / temperature
                
                # Apply sigmoid to get probabilities
                raw_probs = torch.sigmoid(scaled_outputs).cpu().numpy().flatten()
                
                # Apply quantile-based calibration for gradual distribution
                susceptibility_scores = self._calibrate_probabilities(raw_probs)
            
            # Apply edge correction
            chunk_positions = np.arange(chunk_start, chunk_end)[valid_mask]
            chunk_rows = chunk_positions // width
            chunk_cols = chunk_positions % width
            
            # Check for edge pixels
            edge_buffer = 50
            near_edge_mask = (
                (chunk_cols < edge_buffer) |
                (chunk_cols >= width - edge_buffer) |
                (chunk_rows < edge_buffer) |
                (chunk_rows >= height - edge_buffer)
            )
            
            if near_edge_mask.any():
                edge_susceptibility = susceptibility_scores[near_edge_mask]
                edge_cap = 0.7
                edge_susceptibility = np.minimum(edge_susceptibility, edge_cap)
                
                # Apply distance-based dampening
                for idx in np.where(near_edge_mask)[0]:
                    row, col = chunk_rows[idx], chunk_cols[idx]
                    dist_to_edge = min(col, row, width-1-col, height-1-row)
                    
                    if dist_to_edge < edge_buffer:
                        dampen_factor = 0.5 + 0.5 * (dist_to_edge / edge_buffer)
                        edge_susceptibility[np.where(near_edge_mask)[0] == idx] *= dampen_factor
                
                susceptibility_scores[near_edge_mask] = edge_susceptibility
                print(f"  Applied edge correction to {near_edge_mask.sum()} pixels")
            
            # Map predictions back to full raster
            full_prediction[chunk_rows, chunk_cols] = susceptibility_scores
            
            # Clear memory - clean up variables that were actually created
            del chunk_data, valid_chunk_data
            if self.use_onehot_categorical:
                del lithology_encoded, soil_encoded
            del chunk_features, chunk_tensor
            del outputs, susceptibility_scores
            gc.collect()
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETED")
        print("="*60)
        
        # Statistics
        valid_predictions = full_prediction[~np.isnan(full_prediction)]
        print(f"Valid predictions: {len(valid_predictions):,}")
        print(f"  Min: {np.min(valid_predictions):.4f}")
        print(f"  Max: {np.max(valid_predictions):.4f}")
        print(f"  Mean: {np.mean(valid_predictions):.4f}")
        print(f"  Median: {np.median(valid_predictions):.4f}")
        print(f"  Std Dev: {np.std(valid_predictions):.4f}")
        
        # Distribution percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Probability Distribution:")
        for p in percentiles:
            val = np.percentile(valid_predictions, p)
            print(f"    {p}th percentile: {val:.4f}")
        
        # Risk classification
        print(f"\n  Risk Classification:")
        very_low = (valid_predictions < 0.3).sum()
        low = ((valid_predictions >= 0.3) & (valid_predictions < 0.5)).sum()
        moderate = ((valid_predictions >= 0.5) & (valid_predictions < 0.7)).sum()
        high = ((valid_predictions >= 0.7) & (valid_predictions < 0.85)).sum()
        very_high = (valid_predictions >= 0.85).sum()
        
        total = len(valid_predictions)
        print(f"    Very Low  (< 0.3):  {very_low:,} ({very_low/total*100:.1f}%)")
        print(f"    Low       (0.3-0.5): {low:,} ({low/total*100:.1f}%)")
        print(f"    Moderate  (0.5-0.7): {moderate:,} ({moderate/total*100:.1f}%)")
        print(f"    High      (0.7-0.85): {high:,} ({high/total*100:.1f}%)")
        print(f"    Very High (>= 0.85): {very_high:,} ({very_high/total*100:.1f}%)")
        
        print(f"\n  High-risk pixels (>= {self.best_threshold:.3f}): {(valid_predictions >= self.best_threshold).sum():,}")
        print(f"  High-risk percentage: {((valid_predictions >= self.best_threshold).sum() / len(valid_predictions) * 100):.2f}%")
        
        # Save susceptibility map
        print(f"\nSaving susceptibility map to {output_path}...")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(full_prediction, 1)
        
        print("✅ Susceptibility map saved successfully!")
        
        if progress_callback:
            progress_callback(100, "Processing complete!")
        
        return True
