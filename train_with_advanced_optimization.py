#!/usr/bin/env python3
"""
Train a new model with Advanced Threshold Optimization
Creates a model with the new features for testing.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch

# Add the current directory to path 
sys.path.append('/home/anees/Projects/annlandslide_train')

from ann_training_module_improved import ANNTrainingModuleImproved

def train_with_advanced_optimization():
    """Train a model with advanced threshold optimization"""
    
    print("🚀 Training Model with Advanced Threshold Optimization")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ANNTrainingModuleImproved()
        
        # Load training data
        print("📂 Loading training data...")
        X_train_path = "ANN-landslide-susceptibility/data/X_train.csv"
        y_train_path = "ANN-landslide-susceptibility/data/y_train.csv"
        X_val_path = "ANN-landslide-susceptibility/data/X_val.csv"
        y_val_path = "ANN-landslide-susceptibility/data/y_val.csv"
        
        if not all(os.path.exists(p) for p in [X_train_path, y_train_path, X_val_path, y_val_path]):
            print("❌ Training data not found. Please run data preparation first.")
            return False
        
        # Load data
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()
        X_val = pd.read_csv(X_val_path) 
        y_val = pd.read_csv(y_val_path).values.ravel()
        
        # Combine for consistent processing
        X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_combined = np.concatenate([y_train, y_val])
        
        print(f"✅ Data loaded: {len(X_combined)} samples, {X_combined.shape[1]} features")
        print(f"   - Landslides: {np.sum(y_combined)} ({100*np.mean(y_combined):.1f}%)")
        
        # Prepare training data using the trainer's method
        print("🔧 Preparing training data...")
        
        # Create a simple dataframe with features and labels
        training_df = X_combined.copy()
        training_df['label'] = y_combined
        
        # Use trainer's preparation method
        training_data = trainer.prepare_training_data_with_spatial_cv(training_df, test_split=0.2, use_spatial_cv=False)
        
        print("✅ Training data prepared")
        print(f"   - Selected features: {len(training_data['selected_features'])}")
        print(f"   - Training samples: {len(training_data['X_train'])}")
        print(f"   - Test samples: {len(training_data['X_test'])}")
        
        # Train model with advanced optimization
        print("\n🎯 Starting training with advanced threshold optimization...")
        
        result = trainer.train_model(
            training_data=training_data,
            num_epochs=50,  # Reduced for faster testing
            batch_size=64,
            learning_rate=0.001,
            patience=15
        )
        
        print("\n✅ Training completed!")
        
        # Display results
        metrics = result['metrics']
        threshold_opt = result['threshold_optimization']
        
        print("\n📊 TRAINING RESULTS:")
        print("=" * 40)
        print(f"🎯 Optimized Threshold: {result['best_threshold']:.3f}")
        print(f"🏆 Best Method: {threshold_opt['best_method']}")
        print(f"📈 Test F1 Score: {metrics['f1']:.3f}")
        print(f"📈 Test Recall: {metrics['recall']:.3f}")
        print(f"📈 Test Precision: {metrics['precision']:.3f}")
        print(f"📈 Test AUC-ROC: {metrics['auc_roc']:.3f}")
        
        print(f"\n📋 Threshold Options:")
        for method, results in threshold_opt['all_results'].items():
            print(f"   {method}: {results['threshold']:.3f} "
                  f"(F1={results['f1']:.3f}, Recall={results['recall']:.3f})")
        
        # Save model with advanced optimization results
        print("\n💾 Saving model with advanced optimization...")
        
        model_save_path = "outputs/advanced_optimized_model.pth"
        os.makedirs("outputs", exist_ok=True)
        
        # Create comprehensive save dictionary
        model_save_dict = {
            'model_state_dict': result['model'].state_dict(),
            'scaler': result['scaler'],
            'selected_features': result['selected_features'],
            'best_threshold': result['best_threshold'],
            'threshold_optimization': result['threshold_optimization'],
            'input_size': result['input_size'],
            'training_info': {
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses'],
                'metrics': result['metrics'],
                'epochs_trained': len(result['train_losses'])
            }
        }
        
        torch.save(model_save_dict, model_save_path)
        
        print(f"✅ Model saved to: {model_save_path}")
        
        # Generate summary
        print("\n" + "=" * 60)
        print("🎉 ADVANCED THRESHOLD OPTIMIZATION - SUCCESS!")
        print("=" * 60)
        print("✅ Model trained with comprehensive threshold optimization")
        print("✅ Multiple optimization methods tested and compared")
        print("✅ Best threshold automatically selected")
        print("✅ Model calibration attempted")
        print("✅ Results saved with complete optimization data")
        
        print(f"\n🚀 Ready for deployment!")
        print(f"   Model: {model_save_path}")
        print(f"   Optimal threshold: {result['best_threshold']:.3f}")
        print(f"   Expected F1 score: {metrics['f1']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    success = train_with_advanced_optimization()
    
    if success:
        print("\n🎯 Advanced threshold optimization training completed successfully!")
        print("You can now use the optimized model for predictions.")
    else:
        print("\n❌ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()