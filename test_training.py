#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ANN training functionality
Run this to test the training pipeline without QGIS
"""

import os
import sys
import tempfile

def test_sample_data_generation():
    """Test sample data generation"""
    print("Testing sample data generation...")
    
    try:
        from raster_data_extractor import create_sample_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            landslide_path, non_landslide_path = create_sample_data(temp_dir)
            
            # Check files exist
            assert os.path.exists(landslide_path), "Landslide CSV not created"
            assert os.path.exists(non_landslide_path), "Non-landslide CSV not created"
            
            # Check file contents
            import pandas as pd
            
            landslide_df = pd.read_csv(landslide_path)
            non_landslide_df = pd.read_csv(non_landslide_path)
            
            assert len(landslide_df) > 0, "Landslide CSV is empty"
            assert len(non_landslide_df) > 0, "Non-landslide CSV is empty"
            
            print(f"✓ Sample data generated successfully")
            print(f"  - Landslides: {len(landslide_df)} samples")
            print(f"  - Non-landslides: {len(non_landslide_df)} samples")
            print(f"  - Features: {len(landslide_df.columns)} columns")
            
            return landslide_path, non_landslide_path
            
    except ImportError as e:
        print(f"✗ Missing dependencies for sample data generation: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Sample data generation failed: {e}")
        return None, None

def test_csv_training():
    """Test CSV-based training"""
    print("\nTesting CSV-based training...")
    
    try:
        from simple_training_module import simple_train_model_from_csv
        
        # Generate sample data first
        landslide_path, non_landslide_path = test_sample_data_generation()
        
        if not landslide_path or not non_landslide_path:
            print("✗ Cannot test training without sample data")
            return False
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy files to temp directory (since original are in temp dir that may be deleted)
            import pandas as pd
            import shutil
            
            temp_landslide = os.path.join(temp_dir, 'landslides.csv')
            temp_non_landslide = os.path.join(temp_dir, 'non_landslides.csv')
            output_model = os.path.join(temp_dir, 'test_model.pth')
            
            # Copy files
            shutil.copy2(landslide_path, temp_landslide)
            shutil.copy2(non_landslide_path, temp_non_landslide)
            
            def progress_callback(progress, message):
                print(f"  {progress}%: {message}")
                
            # Train with minimal epochs for testing
            result_path = simple_train_model_from_csv(
                landslide_csv_path=temp_landslide,
                non_landslide_csv_path=temp_non_landslide,
                output_model_path=output_model,
                epochs=5,  # Minimal epochs for testing
                batch_size=16,
                test_split=0.2,
                progress_callback=progress_callback
            )
            
            # Check model file was created
            assert os.path.exists(result_path), "Model file not created"
            
            # Try loading the model
            import torch
            model_data = torch.load(result_path, map_location='cpu')
            
            required_keys = ['model_state_dict', 'scaler', 'selected_features', 'training_info']
            for key in required_keys:
                assert key in model_data, f"Missing key in model file: {key}"
                
            print(f"✓ CSV training completed successfully")
            print(f"  - Model saved to: {result_path}")
            print(f"  - Features used: {len(model_data['selected_features'])}")
            print(f"  - Training accuracy: {model_data['training_info'].get('test_accuracy', 'N/A'):.3f}")
            
            return True
            
    except ImportError as e:
        print(f"✗ Missing dependencies for training: {e}")
        print("  Install with: pip install torch scikit-learn pandas numpy")
        return False
    except Exception as e:
        print(f"✗ CSV training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raster_extraction():
    """Test raster data extraction (if rasterio available)"""
    print("\nTesting raster data extraction...")
    
    try:
        from raster_data_extractor import RasterDataExtractor
        print("✓ Raster extraction module available")
        print("  (Note: Actual raster testing requires raster files)")
        return True
    except ImportError as e:
        print(f"✗ Raster extraction not available: {e}")
        print("  Install with: pip install rasterio geopandas")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...")
    
    dependencies = [
        ('pandas', 'pip install pandas'),
        ('numpy', 'pip install numpy'),
        ('sklearn', 'pip install scikit-learn'),
        ('torch', 'pip install torch'),
    ]
    
    optional_dependencies = [
        ('rasterio', 'pip install rasterio'),
        ('geopandas', 'pip install geopandas'),
    ]
    
    all_good = True
    
    for dep, install_cmd in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - Install with: {install_cmd}")
            all_good = False
            
    print("\nOptional dependencies:")
    for dep, install_cmd in optional_dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"○ {dep} - Install with: {install_cmd}")
            
    return all_good

def main():
    """Run all tests"""
    print("=" * 60)
    print("ANN Landslide Training - Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠ Some required dependencies are missing.")
        print("Please install missing packages before running the plugin.")
        return False
        
    print("\n" + "=" * 60)
    
    # Test sample data generation
    success_sample = test_sample_data_generation() != (None, None)
    
    # Test CSV training
    success_training = test_csv_training()
    
    # Test raster extraction
    success_raster = test_raster_extraction()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Sample Data Generation: {'✓ PASS' if success_sample else '✗ FAIL'}")
    print(f"  CSV Training: {'✓ PASS' if success_training else '✗ FAIL'}")
    print(f"  Raster Extraction: {'✓ AVAILABLE' if success_raster else '○ OPTIONAL'}")
    
    if success_sample and success_training:
        print("\n🎉 Core training functionality is working!")
        print("The plugin should work for CSV-based training.")
        if success_raster:
            print("Raster-based training should also work with proper QGIS integration.")
    else:
        print("\n⚠ Some tests failed. Check error messages above.")
        
    return success_sample and success_training

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)