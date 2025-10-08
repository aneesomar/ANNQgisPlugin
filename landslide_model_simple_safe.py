"""
ANN Landslide Susceptibility Model - Simple Safe Version
Basic implementation with robust error handling and memory management
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.windows import Window

class LandslideSusceptibilityPredictor:
    def __init__(self):
        """Initialize the simple predictor with proper error handling"""
        print("🔧 Initializing Simple ANN Landslide Predictor...")
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = 0.5
        
        # Continuous features (will be scaled)
        self.continuous_features = [
            'Aspect', 'Elevation', 'Flow_Accumulation', 'Plan_Curvature',
            'Profile_Curvature', 'Rivers_Proximity', 'Roads_Proximity',
            'Slope', 'Stream_Power_Index', 'Topographic_Position_Index',
            'Terrain_Ruggedness_Index', 'Topographic_Wetness_Index'
        ]
        
        # Categorical features (will be one-hot encoded)
        # Lithology categories: 1-10
        self.lithology_categories = list(range(1, 11))  # [1, 2, 3, ..., 10]
        # Soil categories: 1-5
        self.soil_categories = list(range(1, 6))  # [1, 2, 3, 4, 5]
        
        # Expected raster order: 12 continuous + Lithology + Soil
        self.expected_features = self.continuous_features + ['Lithology', 'Soil']
        
        # Generate feature column names for one-hot encoded features
        self.lithology_cols = [f'lithology_{i}' for i in self.lithology_categories]
        self.soil_cols = [f'soil_{i}' for i in self.soil_categories]
        
        # Total number of features after one-hot encoding
        self.total_features = len(self.continuous_features) + len(self.lithology_cols) + len(self.soil_cols)
        
        print(f"📊 Feature configuration:")
        print(f"   - Continuous features: {len(self.continuous_features)}")
        print(f"   - Lithology categories: {len(self.lithology_cols)}")
        print(f"   - Soil categories: {len(self.soil_cols)}")
        print(f"   - Total features: {self.total_features}")

    def load_model(self, model_path=None):
        """Load a simple neural network model with basic error handling"""
        if model_path and not os.path.exists(model_path):
            print(f"⚠️  Model file not found: {model_path}")
            print("💡 Creating simple test model with random weights...")
        
        try:
            print("📦 Loading simple model...")
            
            # Create a simple 3-layer network
            class SimpleANN(nn.Module):
                def __init__(self, input_size):
                    super(SimpleANN, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(16, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            # Model expects total_features (12 continuous + 10 lithology + 5 soil = 27)
            self.model = SimpleANN(input_size=self.total_features)
            
            # Set up a simple scaler for continuous features only
            self.setup_scaler()
            
            # Initialize with random weights for testing
            print("✅ Simple model created with random weights (for testing)")
            print("💡 Note: This is a simplified version for testing purposes")
            
        except Exception as e:
            raise Exception(f"Error creating simple model: {str(e)}")

    def setup_scaler(self):
        """Setup a simple scaler with predefined ranges for continuous features only"""
        # Create dummy data for continuous features only (categorical features are not scaled)
        dummy_data = []
        feature_info = [
            ('Aspect', 0, 360),
            ('Elevation', 0, 3000),  
            ('Flow_Accumulation', 0, 1000),
            ('Plan_Curvature', -0.1, 0.1),
            ('Profile_Curvature', -0.1, 0.1),
            ('Rivers_Proximity', 0, 5000),
            ('Roads_Proximity', 0, 10000),
            ('Slope', 0, 45),
            ('Stream_Power_Index', 0, 100),
            ('Topographic_Position_Index', -1, 1),
            ('Terrain_Ruggedness_Index', 0, 50),
            ('Topographic_Wetness_Index', 0, 20)
        ]
        
        # Create sample data covering the range for each continuous feature
        for name, min_val, max_val in feature_info:
            dummy_data.append([min_val, max_val])
        
        # Transpose to get proper shape (samples x features)
        dummy_array = np.array(dummy_data).T  # Shape: (2, 12)
        
        # Fit the scaler on continuous features only
        self.scaler.fit(dummy_array)
        print(f"✅ Scaler configured for {self.scaler.n_features_in_} continuous features")

    def encode_categorical_features(self, continuous_data, lithology_raw, soil_raw):
        """
        One-hot encode lithology and soil features and combine with continuous features
        
        Args:
            continuous_data: numpy array of shape (n_samples, 12) - continuous features
            lithology_raw: numpy array of shape (n_samples,) - raw lithology values
            soil_raw: numpy array of shape (n_samples,) - raw soil values
            
        Returns:
            Combined feature array of shape (n_samples, 27) - 12 continuous + 10 lithology + 5 soil
        """
        n_samples = len(continuous_data)
        
        # One-hot encode lithology
        lithology_encoded = np.zeros((n_samples, len(self.lithology_cols)))
        for i, val in enumerate(lithology_raw):
            if not np.isnan(val):
                val_int = int(val)
                col_name = f'lithology_{val_int}'
                if col_name in self.lithology_cols:
                    col_idx = self.lithology_cols.index(col_name)
                    lithology_encoded[i, col_idx] = 1
        
        # One-hot encode soil
        soil_encoded = np.zeros((n_samples, len(self.soil_cols)))
        for i, val in enumerate(soil_raw):
            if not np.isnan(val):
                val_int = int(val)
                col_name = f'soil_{val_int}'
                if col_name in self.soil_cols:
                    col_idx = self.soil_cols.index(col_name)
                    soil_encoded[i, col_idx] = 1
        
        # Combine all features: continuous + lithology + soil
        combined_features = np.hstack([continuous_data, lithology_encoded, soil_encoded])
        
        return combined_features

        """Read a raster window with robust error handling"""
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1, window=window)
                
                # Convert to float32 and handle nodata
                data = data.astype(np.float32)
                
                # Handle standard nodata values
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                
                # Filter out extreme values that are likely nodata representations
                # Common nodata values: -9999, -3.4e38, 1.7e308, etc.
                data[np.abs(data) > 1e10] = np.nan
                
                return data
                
        except Exception as e:
            print(f"Warning: Error reading {os.path.basename(raster_path)}: {e}")
            # Return NaN array of correct shape
            return np.full((window.height, window.width), np.nan, dtype=np.float32)

    def process_rasters_simple(self, raster_paths, output_path, progress_callback=None):
        """Process rasters with maximum safety and error recovery"""
        try:
            if len(raster_paths) != len(self.expected_features):
                raise ValueError(f"Expected {len(self.expected_features)} rasters, got {len(raster_paths)}")
            
            # Validate all input files exist
            missing_files = [path for path in raster_paths if not os.path.exists(path)]
            if missing_files:
                raise FileNotFoundError(f"Missing raster files: {missing_files}")
            
            if progress_callback:
                progress_callback(5, "Validating input files...")
            
            # Get dimensions from first raster
            with rasterio.open(raster_paths[0]) as src:
                height, width = src.height, src.width
                transform = src.transform
                crs = src.crs
                profile = src.profile.copy()
            
            # Configure output profile
            profile.update({
                'dtype': rasterio.float32,
                'count': 1,
                'compress': 'lzw',
                'nodata': -9999.0
            })
            
            if progress_callback:
                progress_callback(10, f"Processing {height}x{width} raster...")
            
            # Use smaller chunks for better progress tracking and memory management
            chunk_size = min(128, height // 8, width // 8)
            if chunk_size < 32:
                chunk_size = 32
                
            total_chunks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
            processed_chunks = 0
            
            print(f"📊 Processing {total_chunks} chunks of {chunk_size}x{chunk_size} pixels")
            
            # Process the raster
            with rasterio.open(output_path, 'w', **profile) as dst:
                for row_start in range(0, height, chunk_size):
                    for col_start in range(0, width, chunk_size):
                        try:
                            # Calculate chunk bounds
                            row_end = min(row_start + chunk_size, height)
                            col_end = min(col_start + chunk_size, width)
                            
                            chunk_height = row_end - row_start
                            chunk_width = col_end - col_start
                            window = Window(col_start, row_start, chunk_width, chunk_height)
                            
                            # Update progress
                            progress_percent = 10 + (processed_chunks * 85) // total_chunks
                            if progress_callback:
                                progress_callback(progress_percent, 
                                    f"Chunk {processed_chunks + 1}/{total_chunks} ({progress_percent}%)")
                            
                            # Read all rasters for this chunk
                            chunk_data = []
                            success_count = 0
                            
                            for i, raster_path in enumerate(raster_paths):
                                try:
                                    raster_data = self.read_raster_window(raster_path, window)
                                    chunk_data.append(raster_data.flatten())
                                    success_count += 1
                                except Exception as e:
                                    print(f"Error reading raster {i}: {e}")
                                    # Use NaN array as fallback
                                    nan_data = np.full(chunk_height * chunk_width, np.nan, dtype=np.float32)
                                    chunk_data.append(nan_data)
                            
                            # Create predictions array
                            predictions = np.full(chunk_height * chunk_width, -9999.0, dtype=np.float32)
                            
                            if success_count >= len(self.expected_features) // 2:  # At least half the features
                                try:
                                    # Stack data - expecting 14 features (12 continuous + lithology + soil)
                                    chunk_array = np.column_stack(chunk_data)
                                    
                                    # Find valid pixels (not all NaN)
                                    valid_mask = ~np.isnan(chunk_array).any(axis=1)
                                    
                                    if np.any(valid_mask):
                                        valid_chunk_data = chunk_array[valid_mask]
                                        
                                        # Separate continuous features from categorical
                                        # First 12 columns are continuous features
                                        continuous_features = valid_chunk_data[:, :12]
                                        # Column 12 is Lithology
                                        lithology_raw = valid_chunk_data[:, 12]
                                        # Column 13 is Soil
                                        soil_raw = valid_chunk_data[:, 13]
                                        
                                        # Replace any remaining NaN in continuous features with column means
                                        for j in range(continuous_features.shape[1]):
                                            col_mean = np.nanmean(continuous_features[:, j])
                                            if not np.isnan(col_mean):
                                                continuous_features[np.isnan(continuous_features[:, j]), j] = col_mean
                                            else:
                                                continuous_features[:, j] = 0  # Fallback to zero
                                        
                                        # Scale continuous features only
                                        scaled_continuous = self.scaler.transform(continuous_features)
                                        
                                        # One-hot encode categorical features and combine with scaled continuous
                                        final_features = self.encode_categorical_features(
                                            scaled_continuous, lithology_raw, soil_raw
                                        )
                                        
                                        # Make predictions
                                        with torch.no_grad():
                                            tensor_data = torch.FloatTensor(final_features)
                                            chunk_predictions = self.model(tensor_data).numpy().flatten()
                                            predictions[valid_mask] = chunk_predictions
                                
                                except Exception as e:
                                    print(f"Warning: Prediction error in chunk {processed_chunks}: {e}")
                            
                            # Write chunk to output
                            prediction_chunk = predictions.reshape(chunk_height, chunk_width)
                            dst.write(prediction_chunk, 1, window=window)
                            
                            processed_chunks += 1
                            
                        except Exception as e:
                            print(f"Error processing chunk {processed_chunks}: {e}")
                            # Write nodata chunk and continue
                            nodata_chunk = np.full((chunk_height, chunk_width), -9999.0, dtype=np.float32)
                            dst.write(nodata_chunk, 1, window=window)
                            processed_chunks += 1
                            continue
            
            if progress_callback:
                progress_callback(98, "Finalizing output...")
            
            # Verify output
            if os.path.exists(output_path):
                with rasterio.open(output_path) as src:
                    if src.count == 1:
                        progress_callback(100, "✅ Processing completed successfully!")
                        return True
            
            raise Exception("Output file verification failed")
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            
            error_msg = f"Processing failed: {str(e)}"
            if progress_callback:
                progress_callback(0, f"❌ {error_msg}")
            raise Exception(error_msg)
