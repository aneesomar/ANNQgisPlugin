#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to test and demonstrate the ANN training functionality
"""

import os
import sys

def demo_complete_workflow():
    """Demonstrate the complete training workflow"""
    print("=" * 70)
    print("ANN Landslide Susceptibility - Training Demo")
    print("=" * 70)
    
    # Step 1: Generate sample data
    print("\n🔧 Step 1: Generating sample training data...")
    
    try:
        from raster_data_extractor import create_sample_data
        
        # Create sample data in current directory
        landslide_path, non_landslide_path = create_sample_data()
        
        print(f"✓ Sample data created:")
        print(f"  📄 Landslides: {landslide_path}")
        print(f"  📄 Non-landslides: {non_landslide_path}")
        
        # Show data preview
        import pandas as pd
        landslide_df = pd.read_csv(landslide_path)
        non_landslide_df = pd.read_csv(non_landslide_path)
        
        print(f"\n📊 Data Summary:")
        print(f"  • Landslide samples: {len(landslide_df)}")
        print(f"  • Non-landslide samples: {len(non_landslide_df)}")
        print(f"  • Features: {len([c for c in landslide_df.columns if c not in ['xcoord', 'ycoord', 'fid']])}")
        print(f"  • Feature names: {', '.join([c for c in landslide_df.columns if c not in ['xcoord', 'ycoord', 'fid']])}")
        
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False
        
    # Step 2: Train the model
    print(f"\n🧠 Step 2: Training ANN model...")
    
    try:
        from simple_training_module import simple_train_model_from_csv
        
        output_model = "demo_trained_model.pth"
        
        def progress_callback(progress, message):
            if progress % 20 == 0 or progress >= 95:  # Only show major progress updates
                print(f"  {progress:3d}%: {message}")
        
        print(f"  🎯 Output model: {output_model}")
        print(f"  ⏱️  Training with reduced epochs for demo...")
        
        # Train with reduced epochs for demo
        result_path = simple_train_model_from_csv(
            landslide_csv_path=landslide_path,
            non_landslide_csv_path=non_landslide_path,
            output_model_path=output_model,
            epochs=20,  # Reduced for demo
            batch_size=32,
            test_split=0.2,
            progress_callback=progress_callback
        )
        
        print(f"✓ Training completed successfully!")
        print(f"  📦 Model saved: {result_path}")
        
        # Step 3: Examine the trained model
        print(f"\n🔍 Step 3: Examining trained model...")
        
        import torch
        model_data = torch.load(result_path, map_location='cpu', weights_only=False)
        
        training_info = model_data.get('training_info', {})
        
        print(f"  📋 Model Details:")
        print(f"    • Architecture: {model_data.get('model_architecture', 'Unknown')}")
        print(f"    • Input features: {model_data.get('input_dim', 'Unknown')}")
        print(f"    • Selected features: {len(model_data.get('selected_features', []))}")
        print(f"    • Training epochs: {training_info.get('epochs_trained', 'Unknown')}")
        print(f"    • Test accuracy: {training_info.get('test_accuracy', 0):.3f}")
        print(f"    • Test F1 score: {training_info.get('test_f1', 0):.3f}")
        
        print(f"\n  🎯 Selected features for training:")
        selected_features = model_data.get('selected_features', [])
        for i, feature in enumerate(selected_features[:10]):  # Show first 10
            print(f"    {i+1:2d}. {feature}")
        if len(selected_features) > 10:
            print(f"    ... and {len(selected_features) - 10} more")
            
        # Step 4: Test prediction
        print(f"\n🔮 Step 4: Testing model prediction...")
        
        # Load test data
        test_data = pd.read_csv(landslide_path).head(5)  # Use first 5 samples
        test_features = test_data[selected_features]
        
        # Apply the same scaling
        scaler = model_data['scaler']
        test_scaled = scaler.transform(test_features)
        
        # Load model for prediction
        from simple_training_module import SimpleLandslideANN
        model = SimpleLandslideANN(len(selected_features))
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Make predictions
        import torch
        with torch.no_grad():
            test_tensor = torch.tensor(test_scaled, dtype=torch.float32)
            outputs = model(test_tensor)
            probabilities = torch.sigmoid(outputs).numpy()
            
        print(f"  📊 Sample predictions:")
        for i, prob in enumerate(probabilities[:5]):
            risk_level = "HIGH" if prob[0] > 0.7 else "MEDIUM" if prob[0] > 0.3 else "LOW"
            print(f"    Sample {i+1}: {prob[0]:.3f} ({risk_level} risk)")
            
        print(f"\n🎉 Demo completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. 📂 Check the generated files: {landslide_path}, {non_landslide_path}")
        print(f"  2. 🧠 Examine the trained model: {result_path}")
        print(f"  3. 🔌 Use the QGIS plugin interface for real-world data")
        print(f"  4. 📖 Read TRAINING_GUIDE.md for detailed usage instructions")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_plugin_integration_info():
    """Show information about plugin integration"""
    print(f"\n" + "=" * 70)
    print("🔌 QGIS Plugin Integration")
    print("=" * 70)
    
    print(f"""
The ANN Landslide Susceptibility plugin now includes BOTH:

📍 Original Functionality:
  • Load pre-trained .pth models
  • Generate landslide susceptibility maps from rasters
  • Advanced neural network predictions

🆕 New Training Functionality:
  • Train new models directly in QGIS
  • Multiple training methods:
    - Train from QGIS raster layers + landslide points
    - Train from pre-processed CSV files  
    - Generate sample data for testing
  • Progress tracking and status updates
  • Advanced feature selection and model architecture

🎯 Plugin Menu Structure:
  Raster → ANN Landslide Susceptibility →
    ├── Run Landslide Susceptibility  (Original prediction)
    └── Train New Model              (New training interface)

📋 Usage Workflow:
  1. Collect environmental raster data (DEM, slope, geology, etc.)
  2. Prepare landslide inventory points
  3. Use "Train New Model" to create custom .pth model
  4. Use "Run Landslide Susceptibility" with your trained model
  5. Generate susceptibility maps for your study area

📁 File Structure:
  • Original prediction: annLandslide_dialog.py
  • New training: comprehensive_training_dialog.py
  • Core training: ann_training_module.py, simple_training_module.py
  • Sample data: raster_data_extractor.py
  • Documentation: TRAINING_GUIDE.md

The plugin is self-contained and provides a complete landslide
susceptibility modeling solution within QGIS!
""")

if __name__ == "__main__":
    # Run the demo
    success = demo_complete_workflow()
    
    # Show integration info
    show_plugin_integration_info()
    
    if success:
        print(f"\n✨ All systems working! The plugin is ready for use.")
    else:
        print(f"\n⚠️  Some issues detected. Check error messages above.")
        
    sys.exit(0 if success else 1)