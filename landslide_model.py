"""
ANN Landslide Susceptibility Model
This module handles the loading and prediction with the trained ANN model
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import gc

class AttentionLayer(nn.Module):
    """Attention mechanism for the ANN model"""
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
    """Residual block for the ANN model"""
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
    """Advanced ANN model for landslide susceptibility prediction"""
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

class LandslideSusceptibilityPredictor:
    """Main class for landslide susceptibility prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.threshold = 0.5
        self.input_dim = None
        self._scaler_fitted = False
        
        # Expected feature order for raster inputs
        self.expected_raster_order = [
            'aspect_utm15_aligned.tif',
            'elv_aligned.tif', 
            'flow_acc_aligned.tif',
            'planCurv_aligned.tif',
            'profCurv_aligned.tif',
            'riversprox_aligned.tif',
            'roadsprox_aligned.tif',
            'slope_aligned.tif',
            'SPI_aligned.tif',
            'TPI_aligned.tif',
            'TRI_aligned.tif',
            'TWI_aligned.tif',
            'lithology_aligned.tif',
            'soil_aligned.tif'
        ]
        
        # Feature columns that need scaling
        self.columns_to_scale = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv',
                                'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI']
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            # Load model data
            model_data = torch.load(model_path, weights_only=False)
            
            if isinstance(model_data, dict):
                if 'model' in model_data:
                    self.model = model_data['model']
                elif 'model_state_dict' in model_data:
                    # Get model architecture info
                    self.input_dim = model_data['input_dim']
                    
                    # Create model and load state dict
                    self.model = AdvancedLandslideANN(self.input_dim)
                    self.model.load_state_dict(model_data['model_state_dict'])
                    
                    # Load threshold if available
                    self.threshold = model_data.get('best_threshold', 0.5)
                    
                    # Get selected features if available
                    if 'selected_features' in model_data:
                        self.selected_features = model_data['selected_features']
                else:
                    raise ValueError(f"Model file structure not recognized. Available keys: {list(model_data.keys())}")
            else:
                self.model = model_data
                
            self.model.eval()
            return True
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def setup_scaler(self, training_data_path=None):
        """Setup the scaler for feature normalization"""
        try:
            # If training data is provided, fit scaler on it
            if training_data_path and os.path.exists(training_data_path):
                training_data = pd.read_csv(training_data_path)
                # Filter to only continuous columns that need scaling
                scale_data = training_data[self.columns_to_scale]
                self.scaler = MinMaxScaler()
                self.scaler.fit(scale_data)
            else:
                # Create a default scaler (will be fitted on first data)
                self.scaler = MinMaxScaler()
                self._scaler_fitted = False
        except Exception as e:
            raise Exception(f"Error setting up scaler: {str(e)}")
    
    def process_rasters(self, raster_paths, output_path, chunk_size=50000, progress_callback=None):
        """Process raster data and generate susceptibility map"""
        try:
            if not self.model:
                raise Exception("Model not loaded. Call load_model() first.")
            
            # Read and stack raster data
            arrays = []
            reference_src = None
            
            for i, path in enumerate(raster_paths):
                if progress_callback:
                    progress_callback(f"Reading raster {i+1}/{len(raster_paths)}: {os.path.basename(path)}")
                
                with rasterio.open(path) as src:
                    if reference_src is None:
                        reference_src = src
                        profile = src.profile.copy()
                    arrays.append(src.read(1))
            
            # Stack arrays
            stacked = np.stack(arrays, axis=0)
            bands, height, width = stacked.shape
            total_pixels = height * width
            
            if progress_callback:
                progress_callback(f"Processing {total_pixels:,} pixels in chunks of {chunk_size:,}")
            
            # Create output array
            full_prediction = np.full((height, width), np.nan, dtype=np.float32)
            
            # Process in chunks
            processed_pixels = 0
            for chunk_start in range(0, total_pixels, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pixels)
                
                # Convert indices to 2D coordinates
                indices = np.unravel_index(range(chunk_start, chunk_end), (height, width))
                
                # Extract pixel values for this chunk
                pixel_data = stacked[:, indices[0], indices[1]].T
                
                # Check for valid pixels (no NaN values)
                valid_mask = ~np.isnan(pixel_data).any(axis=1)
                valid_data = pixel_data[valid_mask]
                
                if len(valid_data) > 0:
                    # Prepare features
                    features_df = self.prepare_features(valid_data)
                    
                    # Make predictions
                    predictions = self.predict(features_df)
                    
                    # Store predictions
                    chunk_predictions = np.full(len(pixel_data), np.nan)
                    chunk_predictions[valid_mask] = predictions
                    
                    # Update output array
                    chunk_indices = (indices[0][valid_mask], indices[1][valid_mask])
                    full_prediction[chunk_indices] = predictions
                
                processed_pixels += (chunk_end - chunk_start)
                if progress_callback:
                    progress = int((processed_pixels / total_pixels) * 100)
                    progress_callback(f"Processed {processed_pixels:,}/{total_pixels:,} pixels ({progress}%)")
            
            # Save output
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(full_prediction, 1)
            
            if progress_callback:
                progress_callback(f"Susceptibility map saved to: {output_path}")
            
            return True
            
        except Exception as e:
            raise Exception(f"Error processing rasters: {str(e)}")
    
    def prepare_features(self, pixel_data):
        """Prepare features from raw pixel data for prediction"""
        # Convert to DataFrame with proper column names
        feature_names = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv',
                        'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI',
                        'lithology', 'soil']
        
        df = pd.DataFrame(pixel_data, columns=feature_names)
        
        # Scale continuous features
        scale_columns = [col for col in self.columns_to_scale if col in df.columns]
        
        if self.scaler is None:
            # Create and fit scaler on current data
            self.scaler = MinMaxScaler()
            df[scale_columns] = self.scaler.fit_transform(df[scale_columns])
            self._scaler_fitted = True
        elif not hasattr(self, '_scaler_fitted') or not self._scaler_fitted:
            # Fit scaler on current data if not already fitted
            df[scale_columns] = self.scaler.fit_transform(df[scale_columns])
            self._scaler_fitted = True
        else:
            # Use already fitted scaler
            df[scale_columns] = self.scaler.transform(df[scale_columns])
        
        # One-hot encode categorical features
        df = self.one_hot_encode_features(df)
        
        # Select features if feature selection was used during training
        if self.selected_features:
            # Ensure all selected features are present
            missing_features = set(self.selected_features) - set(df.columns)
            if missing_features:
                # Add missing features as zeros
                for feature in missing_features:
                    df[feature] = 0
            
            # Select only the features used during training
            df = df[self.selected_features]
        
        return df
    
    def one_hot_encode_features(self, df):
        """One-hot encode categorical features"""
        # Get unique values for categorical features (you may need to adjust these based on your data)
        lithology_values = list(range(1, 15))  # Adjust based on your lithology classes
        soil_values = list(range(1, 12))       # Adjust based on your soil classes
        
        # One-hot encode lithology
        for val in lithology_values:
            df[f'lithology_{val}'] = (df['lithology'] == val).astype(int)
        
        # One-hot encode soil
        for val in soil_values:
            df[f'soil_{val}'] = (df['soil'] == val).astype(int)
        
        # Drop original categorical columns
        df = df.drop(['lithology', 'soil'], axis=1)
        
        return df
    
    def predict(self, features_df):
        """Make predictions using the loaded model"""
        try:
            # Convert to tensor
            X_tensor = torch.FloatTensor(features_df.values)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.sigmoid(outputs).numpy().flatten()
            
            return probabilities
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def get_expected_raster_names(self):
        """Return the list of expected raster file names in order"""
        return self.expected_raster_order.copy()
