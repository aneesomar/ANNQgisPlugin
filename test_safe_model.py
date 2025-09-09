#!/usr/bin/env python3
"""
Test the safe model with small test rasters
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from landslide_model_simple_safe import LandslideSusceptibilityPredictor

def create_test_raster(filename, width=100, height=100, value_range=(0, 100)):
    """Create a small test raster"""
    # Random data within the specified range
    data = np.random.uniform(value_range[0], value_range[1], (height, width)).astype(np.float32)
    
    # Define geospatial properties
    transform = from_bounds(0, 0, width, height, width, height)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': width,
        'height': height,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data, 1)
    
    print(f"Created test raster: {filename} ({width}x{height})")

def test_safe_model():
    """Test the safe model with small test rasters"""
    print("🧪 Testing Safe ANN Model with Small Rasters")
    print("=" * 50)
    
    # Create test directory
    test_dir = "test_rasters"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Feature names and their typical ranges
    features_info = [
        ('Aspect', (0, 360)),
        ('Elevation', (100, 2000)),
        ('Flow_Accumulation', (1, 500)),
        ('Plan_Curvature', (-0.05, 0.05)),
        ('Profile_Curvature', (-0.05, 0.05)),
        ('Rivers_Proximity', (0, 2000)),
        ('Roads_Proximity', (0, 5000)),
        ('Slope', (0, 30)),
        ('Stream_Power_Index', (1, 50)),
        ('Topographic_Position_Index', (-0.5, 0.5)),
        ('Terrain_Ruggedness_Index', (0, 20)),
        ('Topographic_Wetness_Index', (5, 15)),
        ('Lithology', (1, 8)),
        ('Soil', (1, 4))
    ]
    
    # Create small test rasters (50x50 pixels)
    raster_paths = []
    print("📊 Creating test rasters...")
    
    for i, (feature_name, value_range) in enumerate(features_info):
        filename = os.path.join(test_dir, f"{feature_name.lower()}.tif")
        create_test_raster(filename, width=50, height=50, value_range=value_range)
        raster_paths.append(filename)
    
    # Test the model
    print("\\n🔧 Initializing predictor...")
    predictor = LandslideSusceptibilityPredictor()
    
    # This will create a simple model with random weights for testing
    print("📦 Loading model...")
    predictor.load_model("dummy_path")  # The safe model doesn't need an actual file
    
    # Process the test rasters
    output_path = os.path.join(test_dir, "landslide_susceptibility_test.tif")
    
    def progress_callback(progress, message):
        print(f"Progress: {progress:3d}% - {message}")
    
    print("\\n🚀 Processing test rasters...")
    try:
        success = predictor.process_rasters_simple(
            raster_paths, 
            output_path, 
            progress_callback=progress_callback
        )
        
        if success:
            print("\\n✅ SUCCESS! Test completed successfully")
            
            # Check output file
            with rasterio.open(output_path) as src:
                data = src.read(1)
                valid_pixels = np.sum(~np.isnan(data) & (data != -9999))
                print(f"📊 Output raster: {src.width}x{src.height}")
                print(f"📊 Valid predictions: {valid_pixels}/{src.width * src.height} pixels")
                print(f"📊 Min/Max values: {np.nanmin(data):.3f} / {np.nanmax(data):.3f}")
        else:
            print("❌ Test failed!")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\\n🧹 Cleaning up test files...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_safe_model()
