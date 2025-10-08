# ANN Landslide Susceptibility Plugin# ANN Landslide Susceptibility QGIS Plugin



QGIS plugin for landslide susceptibility mapping using Artificial Neural Networks.> **v2.8.0 Update**: New simplified training module for better, gradual predictions! 🎉



## Current Version: v2.9.3A robust QGIS plugin for landslide susceptibility prediction using Artificial Neural Networks (ANN).



**Status**: ✅ Production Ready## 🚀 Quick Start



## Core Files### ⚡ **Recommended: Use the Simplified Training Module**



### Plugin ComponentsThe new v2.8.0 includes a simplified training approach that produces **gradual, realistic predictions** instead of binary clusters:

- `annLandslide.py` - Main plugin controller

- `annLandslide_dialog.py` - Prediction dialog```bash

- `annLandslide_dialog_base.ui` - Prediction UI# Train a model

- `comprehensive_training_dialog.py` - Training dialogpython3 ann_training_module_simple.py

- `model_training_dialog_base.ui` - Training UI

- `ann_training_module_improved.py` - Training engine (with auto-balancing)# Test predictions (fast ~10 seconds)

- `landslide_model_improved.py` - Prediction enginepython3 quick_test_simple.py

- `metadata.txt` - Plugin metadata

- `icon.png` - Plugin icon# Make full predictions

- `__init__.py` - Plugin initializerpython3 -c "

- `i18n/` - Translationsfrom landslide_model_simple import LandslideModelSimple

model = LandslideModelSimple('landslide_model_simple.pth')

### Utilitiesmodel.predict(raster_paths=['slope.tif', ...], output_path='susceptibility.tif')

- `diagnose_model.py` - Analyzes trained models"

- `simple_fast_test.py` - Quick validation tool (1 minute)```

- `ultra_fast_test.py` - Ultra-fast sampling tests (2-10 seconds)

- `validate_model.py` - Comprehensive ML metrics**Why simplified?** The previous complex model (v2.7.0) was overfitting, producing binary predictions (0 or 1). The simplified model generalizes better and produces natural gradual probabilities.

- `raster_data_extractor.py` - Raster processing helper

📖 **Read More**: 

## Installation- [VERSION_2.8.0_SUMMARY.md](VERSION_2.8.0_SUMMARY.md) - Complete overview

- [SIMPLIFIED_TRAINING_GUIDE.md](SIMPLIFIED_TRAINING_GUIDE.md) - Usage guide

### For QGIS Users- [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) - Technical details

1. Install from ZIP: `releases/ANNLandslidePlugin_v2.9.3.zip`

2. QGIS → Plugins → Install from ZIP---

3. Enable plugin

## 🎯 Features

### For Developers

```bash- **Simplified Architecture**: Better generalization, no overfitting

# Link to QGIS plugins folder- **Gradual Predictions**: Natural probability distributions (not binary!)

ln -s /home/anees/Projects/annlandslide_train ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ANNLandslidePlugin- **Spatial Cross-Validation**: K-means blocking with 5% buffer zones

```- **Ensemble Feature Selection**: SelectKBest + Random Forest + RFE voting

- **Smart Categorical Handling**: One-hot encoding for lithology and soil

## Key Features (v2.9.3)- **Memory Efficient**: Chunk-based processing (150k pixels per chunk)

- **Fast**: 3-5 minutes for full prediction on typical datasets

### Training- **No Calibration Needed**: Model produces good distributions naturally

- ✅ Automatic test set balancing (fixes spatial clustering issues)

- ✅ Class weighting (handles imbalanced datasets)---

- ✅ Simplified 4-layer architecture (prevents overfitting)

- ✅ Dropout 0.6 + learning rate 0.0001## 📁 Project Structure

- ✅ Early stopping (prevents overfitting)

- ✅ One-hot encoding for categorical features### **Training Modules (Use These!)**

- ✅ Spatial cross-validation

- **`ann_training_module_simple.py`** ⭐ - **RECOMMENDED**: Simplified training (v2.8.0)

### Prediction  - 4-layer architecture (256 → 128 → 64 → 1)

- ✅ Fast prediction (274k pixels/second)  - Higher dropout (0.5) for better generalization

- ✅ Gradual susceptibility maps (Std 0.15-0.30)  - Label smoothing (0.1) prevents overconfidence

- ✅ Handles categorical features automatically  - Produces gradual predictions ✅

- ✅ Edge correction for boundary artifacts

- **`landslide_model_simple.py`** ⭐ - **RECOMMENDED**: Prediction with simplified models

### Validation Tools  - No calibration needed

- `diagnose_model.py` - Check model weights, bias, training info  - Clean, fast predictions

- `simple_fast_test.py` - Full validation in ~1 minute  - Matches training module

- `ultra_fast_test.py` - Quick sampling for rapid testing

- **`quick_test_simple.py`** - Quick testing (5k pixels, ~10 seconds)

## Quick Start

### **Legacy/Alternative Training Modules**

### 1. Train Model

```python- `ann_training_module_improved.py` - Complex model (v2.7.0, overfitting issue)

# In QGIS Python Console- `landslide_model_improved.py` - Prediction with complex models

from ANNLandslidePlugin import comprehensive_training_dialog- `ann_training_module.py` - Original advanced training

# Use GUI to train from rasters or CSV- `simple_training_module.py` - Earlier simplified version

```- `csv_only_training.py` - Minimal dependency training



### 2. Validate Model### **Core Plugin Files**

```bash

python3 diagnose_model.py /path/to/model.pth- `annLandslide.py` - Main plugin class

```- `annLandslide_dialog.py` - Main dialog interface

- `comprehensive_training_dialog.py` - Training interface

### 3. Run Prediction- `__init__.py` - Plugin initialization

```python- `metadata.txt` - Plugin metadata (v2.8.0)

# In QGIS- `icon.png` - Plugin icon

# Plugins → ANN Landslide → Prediction

# Load model, select rasters, run### **UI Files**

```

- `annLandslide_dialog_base.ui` - Main dialog UI

## Expected Results- `model_training_dialog_base.ui` - Training dialog UI

- `model_training_dialog.py` - Training dialog controller

### Good Model (v2.9.3)

- Std Dev: 0.15-0.30 ✅### **Utilities**

- Mean: 0.40-0.70 ✅

- Distribution: Mix of low/moderate/high ✅- `raster_data_extractor.py` - Raster processing utilities

- High-risk: 30-60% ✅- `check_model_scaler.py` - Diagnostic tool for model inspection

- Test set: Balanced (~25% landslides) ✅

├── 📦 Packages

### Bad Model (old versions)

- Std Dev: < 0.05 ❌ (binary predictions)### **Prediction Module**│   └── annlandslide_v2.1.zip         # ⭐ Ready-to-install ZIP package

- Mean: > 0.85 or < 0.15 ❌ (biased)

- Distribution: >95% one category ❌- `landslide_model_simple_safe.py` - Model prediction and mapping│

- High-risk: >95% ❌

- Test set: Imbalanced (>60% landslides) ❌├── 🎯 Models



## Architecture### **Legacy/Testing Files**│   └── landslide_model_advanced_complete.pth # Pre-trained model



### SimpleLandslideANN (4 layers)- `modelTraining.py` - Original training script│

```

Input (n features) → 256 → 128 → 64 → Output (1)- `demo_training.py` - Demo/testing script├── 🌍 Data & Examples

                    ↓     ↓     ↓

                  ReLU  ReLU  ReLU- `test_training.py` - Test utilities│   ├── durbanRasters/                # Sample input rasters

                    ↓     ↓     ↓

               Dropout Dropout Dropout│   ├── outputs/                      # Sample outputs

                 (0.6)  (0.6)  (0.6)

```### **Sample Data**│   └── examples/                     # Example scripts



**Why simplified?**- `durbanRasters/` - Complete raster dataset for testing│

- Prevents overfitting on small datasets (5k-50k samples)

- Produces gradual predictions (not binary)- `models/` - Pre-trained models├── 🌍 Internationalization

- 3x faster than complex architectures

- Better generalization│   └── i18n/                         # Translation files



## Troubleshooting### **Internationalization**│



### "Everything high susceptibility"- `i18n/af.ts` - Translation file└── 📋 Configuration & Installation

→ Retrain with v2.9.3 (auto-balances test set)

    ├── install.sh                    # Installation script

### "Binary predictions (Std < 0.05)"

→ Check if using old model, retrain with v2.9.3## 🚀 **Installation**    ├── create_zip_package.sh         # ZIP package creator



### "KeyError: pandas indexing"1. Install plugin from `annlandslide_FIXED_v2.zip`    ├── requirements.txt               # Dependencies

→ Update to latest v2.9.3 (fixed in current release)

2. Install dependencies: `torch`, `scikit-learn`, `pandas`, `numpy`    ├── QGIS_INSTALLATION_GUIDE.md    # Installation guide

### "Stripe through map"

→ Check raster NoData values, ensure all rasters cover same extent3. Test with sample data    ├── QGIS_RELOAD_INSTRUCTIONS.md   # Reload instructions



## Development    └── README.md                     # This file



### Project Structure## 🎉 **Key Features**```

```

/home/anees/Projects/annlandslide_train/- ✅ **Automated raster sampling** from vector points

├── Core plugin files (14 files)

├── releases/ (ZIP packages for distribution)- ✅ **Multiple training fallbacks** (QGIS → CSV-only → rasterio)## 🔧 Installation

├── models/ (trained .pth files)

├── i18n/ (translations)- ✅ **CPU-only processing** (no CUDA issues)

└── _archive_old_files/ (old versions, docs)

```- ✅ **Sample data generation** for testing### Option 1: Easy ZIP Installation (Recommended)



### Testing Workflow- ✅ **Complete landslide susceptibility mapping**1. Download the plugin package: `packages/annlandslide_v2.1.zip`

```bash

# 1. Diagnose model2. Open QGIS

python3 diagnose_model.py model.pth

## 🔄 **Workflow**3. Go to **Plugins** → **Manage and Install Plugins**

# 2. Quick test

python3 ultra_fast_test.py model.pth --mode quick**Input:** Raster layers + Landslide points → **Output:** Trained .pth model4. Click **"Install from ZIP"**



# 3. Full validation5. Select the downloaded `annlandslide_v2.1.zip` file

python3 simple_fast_test.py model.pth [rasters...]

```## 📋 **Backup**6. Click **"Install Plugin"**



## ReleasesFull project backup saved in: `../annlandslide_backup/`7. Enable the plugin in the plugins list



Latest: **v2.9.3** (2025-10-08)### Option 2: Manual Installation

- Auto-balances imbalanced test sets1. Run the installation script: `./install.sh`

- Fixes spatial clustering issues2. Restart QGIS

- Produces realistic gradual predictions3. Enable the plugin in **Plugins** → **Manage and Install Plugins**



See `releases/` folder for all versions.> **Note**: The installation script automatically copies all necessary files to your QGIS plugins directory.



## License## 📊 Required Input Data



[Add your license here]The plugin requires 14 raster layers in the following order:



## Author1. **Aspect** - Slope aspect (0-360°)

2. **Elevation** - Digital elevation model

Anees Omar3. **Flow Accumulation** - Water flow accumulation

4. **Plan Curvature** - Horizontal curvature

## Support5. **Profile Curvature** - Vertical curvature  

6. **Rivers Proximity** - Distance to rivers

For issues or questions, check:7. **Roads Proximity** - Distance to roads

1. `diagnose_model.py` output8. **Slope** - Slope gradient (0-90°)

2. Training log for rebalancing messages9. **Stream Power Index** - Erosive power of flowing water

3. `releases/ULTIMATE_FIX_v2.9.3.md` for detailed fix info10. **Topographic Position Index** - Relative topographic position

11. **Terrain Ruggedness Index** - Surface roughness
12. **Topographic Wetness Index** - Wetness accumulation
13. **Lithology** - Rock/soil type (categorical)
14. **Soil** - Soil type (categorical)

## 🎯 Usage

1. Open the plugin from the QGIS toolbar
2. Select your 14 input raster files
3. Choose an output location
4. Click "Run Prediction"
5. Monitor progress in real-time
6. View results in QGIS

## 📈 Output

- **Format**: GeoTIFF (.tif)
- **Values**: Probability (0.0 - 1.0)
- **Interpretation**:
  - 0.0-0.2: Very Low Susceptibility
  - 0.2-0.4: Low Susceptibility  
  - 0.4-0.6: Moderate Susceptibility
  - 0.6-0.8: High Susceptibility
  - 0.8-1.0: Very High Susceptibility

## ⚙️ Technical Details

- **Processing**: Single-threaded, chunk-based
- **Chunk Size**: Adaptive (32-128 pixels)
- **Memory Usage**: Optimized for stability
- **Error Handling**: Comprehensive with fallbacks
- **Compression**: LZW compression for outputs

## 🔍 Version Information

- **Current Version**: 2.1 (Safe Version)
- **QGIS Compatibility**: 3.0+
- **Status**: Stable and tested

## 📝 Notes

- This is a simplified version prioritizing stability over performance
- Uses a basic neural network for demonstration purposes
- For production use, replace with a properly trained model
- All processing is CPU-based (no GPU acceleration required)

---

**Status**: ✅ Ready for production use  
**Maintainer**: ANNQgisPlugin Project  
**License**: Open Source
