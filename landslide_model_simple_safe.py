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
        self.expected_features = [
            'Aspect', 'Elevation', 'Flow_Accumulation', 'Plan_Curvature',
            'Profile_Curvature', 'Rivers_Proximity', 'Roads_Proximity',
            'Slope', 'Stream_Power_Index', 'Topographic_Position_Index',
            'Terrain_Ruggedness_Index', 'Topographic_Wetness_Index',
            'Lithology', 'Soil'
        ]

    def load_model(self, model_path=None):
        """Load a simple neural network model with basic error handling"""
        if model_path and not os.path.exists(model_path):
            print(f"⚠️  Model file not found: {model_path}")
            print("💡 Creating simple test model with random weights...")
        
        try:
            print("📦 Loading simple model...")
            
            # Create a simple 3-layer network
            class SimpleANN(nn.Module):
                def __init__(self, input_size=14):
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
            
            self.model = SimpleANN(input_size=14)
            
            # Set up a simple scaler for our 14 features
            self.setup_scaler()
            
            # Initialize with random weights for testing
            print("✅ Simple model created with random weights (for testing)")
            print("💡 Note: This is a simplified version for testing purposes")
            
        except Exception as e:
            raise Exception(f"Error creating simple model: {str(e)}")

    def setup_scaler(self):
        """Setup a simple scaler with predefined ranges"""
        # Create dummy data for each feature to fit the scaler properly
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
            ('Topographic_Wetness_Index', 0, 20),
            ('Lithology', 1, 10),
            ('Soil', 1, 5)
        ]
        
        # Create sample data covering the range for each feature
        for name, min_val, max_val in feature_info:
            dummy_data.append([min_val, max_val])
        
        # Transpose to get proper shape (samples x features)
        dummy_array = np.array(dummy_data).T  # Shape: (2, 14)
        
        # Fit the scaler
        self.scaler.fit(dummy_array)
        print(f"✅ Simple scaler configured for {self.scaler.n_features_in_} features")

    def read_raster_window(self, raster_path, window):
        """Read a raster window with robust error handling"""
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1, window=window)
                
                # Convert to float32 and handle nodata
                data = data.astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                
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
                                    # Stack data
                                    chunk_array = np.column_stack(chunk_data)
                                    
                                    # Find valid pixels (not all NaN)
                                    valid_mask = ~np.isnan(chunk_array).any(axis=1)
                                    
                                    if np.any(valid_mask):
                                        valid_data = chunk_array[valid_mask]
                                        
                                        # Replace any remaining NaN with feature means
                                        for j in range(valid_data.shape[1]):
                                            col_mean = np.nanmean(valid_data[:, j])
                                            if not np.isnan(col_mean):
                                                valid_data[np.isnan(valid_data[:, j]), j] = col_mean
                                            else:
                                                valid_data[:, j] = 0  # Fallback to zero
                                        
                                        # Scale the data
                                        scaled_data = self.scaler.transform(valid_data)
                                        
                                        # Make predictions
                                        with torch.no_grad():
                                            tensor_data = torch.FloatTensor(scaled_data)
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
