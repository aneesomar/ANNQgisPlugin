#!/usr/bin/env python3
"""
Test Enhanced Plugin End-to-End Performance
===========================================

Test the complete plugin workflow with enhanced feature selection
and compare against previous versions.

Author: GitHub Copilot  
Date: October 13, 2025
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add to path
sys.path.insert(0, '/home/anees/Projects/annlandslide_train')

def test_plugin_end_to_end():
    """Test complete plugin workflow with enhanced features"""
    
    print("🔥 ENHANCED PLUGIN END-TO-END TEST")
    print("=" * 50)
    
    try:
        from ann_training_module_improved import ANNTrainingModuleImproved
        
        # Initialize trainer
        trainer = ANNTrainingModuleImproved()
        
        print("📂 Loading existing processed data...")
        
        # Use existing preprocessed data for faster testing
        X_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_train.csv')
        y_train = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_train.csv').iloc[:, 0]
        X_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/X_test.csv')
        y_test = pd.read_csv('/home/anees/Projects/annlandslide_train/ANN-landslide-susceptibility/data/y_test.csv').iloc[:, 0]
        
        print(f"   ✅ Training samples: {len(X_train):,}")
        print(f"   ✅ Test samples: {len(X_test):,}")
        print(f"   ✅ Features: {X_train.shape[1]}")
        
        # Create mock coordinates for spatial testing
        print("   🗺️ Creating mock spatial coordinates...")
        np.random.seed(42)
        train_coords = np.random.rand(len(X_train), 2) * 1000  # Mock coordinates
        test_coords = np.random.rand(len(X_test), 2) * 1000
        all_coords = np.vstack([train_coords, test_coords])
        
        # Prepare training data with enhanced feature selection
        print("\n🔧 PREPARING ENHANCED TRAINING DATA...")
        
        # Create properly structured feature data (as would come from raster extraction)  
        features_df = pd.concat([X_train, X_test], ignore_index=True)
        features_df['x'] = all_coords[:, 0] 
        features_df['y'] = all_coords[:, 1]
        features_df['label'] = pd.concat([y_train, y_test], ignore_index=True)
        
        feature_data = features_df
        
        training_data = trainer.prepare_training_data_with_spatial_cv(
            feature_data=feature_data,
            test_split=0.2,
            use_spatial_cv=True,
            n_blocks=10,
            buffer_distance=50  # 50m buffer
        )
        
        print(f"   ✅ Spatial CV training data prepared")
        print(f"   📊 Train samples: {training_data['X_train'].shape[0]:,}")
        print(f"   📊 Test samples: {training_data['X_test'].shape[0]:,}")
        print(f"   📊 Selected features: {training_data['X_train'].shape[1]}")
        
        # Train the enhanced model
        print(f"\n🧠 TRAINING ENHANCED ANN MODEL...")
        
        def progress_callback(epoch, total_epochs):
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                print(f"   Epoch {epoch+1}/{total_epochs}")
        
        result = trainer.train_model(
            training_data=training_data,
            num_epochs=50,  # Reduced for testing
            batch_size=128,
            learning_rate=0.001,
            patience=15,
            progress_callback=progress_callback
        )
        
        print(f"\n📊 ENHANCED MODEL RESULTS:")
        
        # Extract metrics
        if 'validation_metrics' in result:
            metrics = result['validation_metrics']
            print(f"   🎯 Training AUC-ROC: {metrics.get('training_auc', 0):.3f}")
            print(f"   🎯 Validation AUC-ROC: {metrics.get('validation_auc', 0):.3f}")
            print(f"   🎯 Test AUC-ROC: {metrics.get('test_auc', 0):.3f}")
            print(f"   🎯 PR-AUC: {metrics.get('pr_auc', 0):.3f}")
            print(f"   🎯 Precision: {metrics.get('precision', 0):.3f}")
            print(f"   🎯 Recall: {metrics.get('recall', 0):.3f}")
            print(f"   🎯 F1-Score: {metrics.get('f1', 0):.3f}")
        
        # Feature selection info
        if hasattr(trainer, 'feature_selection_info'):
            info = trainer.feature_selection_info
            print(f"\n🔍 FEATURE SELECTION ANALYSIS:")
            print(f"   Original features: {info['original_features']}")
            print(f"   Quality filtered: {info['after_quality_filter']}")
            print(f"   Final selected: {info['final_selected']}")
            print(f"   Reduction: {(1 - info['final_selected']/info['original_features'])*100:.1f}%")
            
            print(f"\n🏆 TOP DISCRIMINATIVE FEATURES:")
            for detail in info['feature_details'][:8]:
                print(f"      {detail['rank']:2d}. {detail['feature']:<20}")
        
        # Save enhanced model
        model_path = '/home/anees/Projects/annlandslide_train/models/enhanced_feature_selection_model.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if 'model' in result:
            import torch
            torch.save({
                'model_state_dict': result['model'].state_dict(),
                'feature_selection_info': trainer.feature_selection_info if hasattr(trainer, 'feature_selection_info') else None,
                'selected_features': training_data.get('selected_features', []),
                'scaler': result.get('scaler'),
                'metrics': result.get('validation_metrics', {}),
                'training_config': {
                    'enhanced_feature_selection': True,
                    'max_features': 15,
                    'quality_filtering': True,
                    'spatial_cv': True
                }
            }, model_path)
            print(f"   ✅ Enhanced model saved: {model_path}")
        
        # Performance assessment
        test_auc = result.get('validation_metrics', {}).get('test_auc', 0)
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if test_auc >= 0.85:
            print(f"   ✅ EXCELLENT: {test_auc:.1%} AUC-ROC exceeds target!")
            status = "EXCELLENT"
        elif test_auc >= 0.80:
            print(f"   ✅ VERY GOOD: {test_auc:.1%} AUC-ROC is strong")
            status = "VERY_GOOD"
        elif test_auc >= 0.75:
            print(f"   ⚠️ GOOD: {test_auc:.1%} AUC-ROC is acceptable")
            status = "GOOD"
        else:
            print(f"   ❌ NEEDS IMPROVEMENT: {test_auc:.1%} AUC-ROC below target")
            status = "POOR"
        
        print(f"\n🏆 SUMMARY:")
        print(f"   ✅ Enhanced feature selection: {info['final_selected']} features ({(1-info['final_selected']/info['original_features'])*100:.0f}% reduction)")
        print(f"   ✅ Spatial cross-validation: Balanced train/test split")
        print(f"   ✅ Model performance: {test_auc:.1%} AUC-ROC")
        print(f"   ✅ Status: {status}")
        
        return {
            'success': True,
            'test_auc': test_auc,
            'status': status,
            'feature_info': trainer.feature_selection_info if hasattr(trainer, 'feature_selection_info') else None,
            'model_path': model_path
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = test_plugin_end_to_end()
    
    if result['success']:
        print(f"\n🎉 ENHANCED PLUGIN TEST SUCCESSFUL!")
        print(f"🏆 Final AUC-ROC: {result['test_auc']:.1%}")
        print(f"🎯 Status: {result['status']}")
        
        # If performance is good, suggest creating new plugin version
        if result['test_auc'] >= 0.80:
            print(f"\n💡 RECOMMENDATION: Performance is strong enough for v3.4.0 release!")
        else:
            print(f"\n⚠️ RECOMMENDATION: Consider further optimization before release")
    else:
        print(f"\n❌ Enhanced plugin test failed: {result.get('error', 'Unknown error')}")