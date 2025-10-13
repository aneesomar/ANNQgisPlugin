#!/usr/bin/env python3
"""
Quick Model Diagnostics
Analyzes model architecture, weights, and training state
"""

import torch
import sys
import numpy as np

def diagnose_model(model_path):
    print("=" * 70)
    print("MODEL DIAGNOSTICS")
    print("=" * 70)
    
    # Load model
    print(f"\n📁 Loading model: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check checkpoint structure
    print("\n📊 Checkpoint Contents:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Get model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n🏗️  Model Architecture:")
        for name, param in state_dict.items():
            print(f"  {name:40s} {str(param.shape):20s} {param.numel():>10,} params")
        
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n  Total Parameters: {total_params:,}")
    
    # Check for training info
    if 'epoch' in checkpoint:
        print(f"\n📈 Training Info:")
        print(f"  Epoch: {checkpoint['epoch']}")
    
    if 'train_loss' in checkpoint:
        print(f"  Training Loss: {checkpoint['train_loss']:.6f}")
    
    if 'val_loss' in checkpoint:
        print(f"  Validation Loss: {checkpoint['val_loss']:.6f}")
    
    # Analyze output layer bias
    print("\n🎯 Output Layer Analysis:")
    if 'model_state_dict' in checkpoint:
        # Find output layer
        output_weight = None
        output_bias = None
        
        for name, param in state_dict.items():
            if 'output_layer.weight' in name or 'network.6.weight' in name:
                output_weight = param
                print(f"  Output Weight: {name}")
                print(f"    Shape: {output_weight.shape}")
                print(f"    Mean: {output_weight.mean().item():.6f}")
                print(f"    Std: {output_weight.std().item():.6f}")
                print(f"    Min: {output_weight.min().item():.6f}")
                print(f"    Max: {output_weight.max().item():.6f}")
            
            if 'output_layer.bias' in name or 'network.6.bias' in name:
                output_bias = param
                print(f"\n  Output Bias: {name}")
                print(f"    Value: {output_bias.item():.6f}")
                
                # Convert bias to probability (sigmoid)
                prob = torch.sigmoid(output_bias).item()
                print(f"    Sigmoid(bias): {prob:.6f} ({prob*100:.2f}%)")
                
                if prob > 0.8:
                    print("    ⚠️  WARNING: High bias! Model will predict high susceptibility")
                elif prob < 0.2:
                    print("    ⚠️  WARNING: Low bias! Model will predict low susceptibility")
                else:
                    print("    ✅ Bias looks reasonable")
    
    # Check all weights for extreme values
    print("\n⚖️  Weight Distribution:")
    if 'model_state_dict' in checkpoint:
        all_weights = []
        for name, param in state_dict.items():
            if 'weight' in name:
                all_weights.extend(param.flatten().tolist())
        
        all_weights = np.array(all_weights)
        print(f"  Mean: {all_weights.mean():.6f}")
        print(f"  Std: {all_weights.std():.6f}")
        print(f"  Min: {all_weights.min():.6f}")
        print(f"  Max: {all_weights.max():.6f}")
        print(f"  |Weight| > 1: {(np.abs(all_weights) > 1).sum():,} / {len(all_weights):,} ({(np.abs(all_weights) > 1).sum()/len(all_weights)*100:.2f}%)")
        print(f"  |Weight| > 5: {(np.abs(all_weights) > 5).sum():,} / {len(all_weights):,} ({(np.abs(all_weights) > 5).sum()/len(all_weights)*100:.2f}%)")
        
        if (np.abs(all_weights) > 5).sum() > len(all_weights) * 0.01:
            print("  ⚠️  WARNING: Many extreme weights detected! Possible overfitting")
    
    print("\n" + "=" * 70)
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    if output_bias is not None:
        prob = torch.sigmoid(output_bias).item()
        if prob > 0.8:
            print("""
  ⚠️  PROBLEM: Output bias is very high ({:.2f}%)
  
  This means the model learned to predict high susceptibility for everything!
  
  Solutions:
  1. Reduce learning rate (try 0.0001 instead of 0.001)
  2. Increase label smoothing (try 0.2 instead of 0.1)
  3. Add class weights to balance training data:
     - If 38.9% landslides: weight = 1.57 for non-landslides, 0.63 for landslides
  4. Train for fewer epochs (stop early when validation loss stops improving)
  5. Increase dropout to 0.6 or 0.7
  
  Try retraining with these settings!
            """.format(prob * 100))
        elif prob < 0.2:
            print("""
  ⚠️  PROBLEM: Output bias is very low ({:.2f}%)
  
  This means the model learned to predict low susceptibility for everything!
  
  Solutions:
  1. Check if training data is imbalanced (too few landslides)
  2. Add class weights to balance training data
  3. Reduce label smoothing (try 0.05 instead of 0.1)
            """.format(prob * 100))
        else:
            print("""
  ✅ Output bias looks reasonable ({:.2f}%)
  
  If predictions are still bad, the issue might be:
  1. Feature preprocessing/normalization
  2. Input data quality
  3. Model complexity vs dataset size
            """.format(prob * 100))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_model.py [model_path]")
        print("\nExample:")
        print("  python diagnose_model.py /path/to/output5.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    diagnose_model(model_path)
