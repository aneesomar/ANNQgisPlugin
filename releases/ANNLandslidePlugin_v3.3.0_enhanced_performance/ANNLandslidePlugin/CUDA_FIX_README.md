## CUDA Compatibility Fix for ANNLandslidePlugin v3.2.0

### Issue Fixed
- **CUDA Error**: "no kernel image is available for execution on the device"
- **Problem**: PyTorch CUDA compilation incompatibility with some GPU architectures
- **Error Location**: During model training in `ann_training_module_improved.py`

### Solution Applied
1. **Forced CPU Usage**: Modified both training and prediction modules to use CPU instead of CUDA
2. **Disabled Mixed Precision**: Removed CUDA-specific mixed precision training features
3. **Updated Files**:
   - `ann_training_module_improved.py` - Line 348: Changed to `torch.device('cpu')`
   - `landslide_model_improved.py` - Line 151: Changed to `torch.device('cpu')`

### Changes Made
```python
# Before (problematic)
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# After (fixed)
self.device = torch.device('cpu')  # Force CPU for compatibility
```

### Performance Impact
- **Training Speed**: CPU training is slower than GPU, but more compatible
- **Memory Usage**: Lower GPU memory usage, higher RAM usage
- **Compatibility**: Works on all systems regardless of CUDA/GPU setup
- **Reliability**: Eliminates CUDA version conflicts and driver issues

### Alternative Solutions (Advanced Users)
If you want to use GPU acceleration, you can:

1. **Update PyTorch**: Install a PyTorch version compiled for your specific CUDA version
2. **Check CUDA Compatibility**: Ensure your GPU architecture is supported
3. **Environment Variables**: Try setting `CUDA_LAUNCH_BLOCKING=1` for debugging

### When to Use GPU vs CPU
- **Use CPU** (Current Default): 
  - Maximum compatibility
  - Smaller datasets (< 100k samples)
  - Systems without dedicated GPU
  
- **Use GPU** (Manual Enable):
  - Large datasets (> 100k samples)
  - High-end NVIDIA GPUs with proper drivers
  - CUDA-compatible PyTorch installation

### Re-enabling GPU (Advanced)
To re-enable GPU support, change line 348 in `ann_training_module_improved.py` back to:
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Note**: Only do this if you have confirmed CUDA compatibility.

---

**Fix Date**: October 13, 2025  
**Affected Files**: `ann_training_module_improved.py`, `landslide_model_improved.py`  
**Impact**: Training now works reliably on all systems