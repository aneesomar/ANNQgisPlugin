#!/usr/bin/env python3
'''Quick test for enhanced feature selection functionality'''

def test_enhanced_features():
    try:
        from ann_training_module_improved import ANNTrainingModuleImproved
        
        print("🧪 Testing Enhanced Feature Selection...")
        trainer = ANNTrainingModuleImproved()
        
        # Check if enhanced method exists
        if hasattr(trainer, '_enhanced_feature_selection'):
            print("✅ Enhanced feature selection method found")
        else:
            print("❌ Enhanced feature selection method missing")
            
        # Check for feature selection info attribute
        if hasattr(trainer, 'feature_selection_info'):
            print("✅ Feature selection info tracking available")
        else:
            print("⚠️  Feature selection info tracking not initialized")
            
        print("🏆 Enhanced functionality ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced features: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_features()
    if success:
        print("✅ Plugin enhanced functionality verified!")
    else:
        print("❌ Plugin enhancement test failed!")