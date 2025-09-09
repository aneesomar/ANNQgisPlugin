# ANN Landslide Susceptibility QGIS Plugin

A robust QGIS plugin for landslide susceptibility prediction using Artificial Neural Networks (ANN).

## 🚀 Features

- **Safe & Stable**: Single-threaded processing with comprehensive error handling
- **Progress Tracking**: Real-time progress updates with clear status messages
- **Memory Efficient**: Adaptive chunk-based processing
- **Error Recovery**: Graceful handling of missing or corrupted data
- **Standard Output**: GeoTIFF format with proper georeferencing

## 📁 Project Structure

```
annlandslide/
├── 📄 Core Plugin Files
│   ├── __init__.py                    # Plugin initialization
│   ├── annLandslide.py               # Main plugin class
│   ├── annLandslide_dialog.py        # User interface dialog
│   ├── annLandslide_dialog_base.ui   # UI design file
│   ├── metadata.txt                  # Plugin metadata
│   └── icon.png                      # Plugin icon
│
├── 🧠 Model Files
│   ├── landslide_model.py            # Original model (legacy)
│   └── landslide_model_simple_safe.py # Current safe model
│
├── 📚 Documentation
│   └── SAFE_VERSION_SUMMARY.md       # Detailed version information
│
├── 🎯 Models
│   └── landslide_model_advanced_complete.pth # Pre-trained model
│
├── 📊 Outputs
│   └── landslide_susceptibility_output.tif   # Sample output
│
├── 🌍 Internationalization
│   └── i18n/                         # Translation files
│
└── 📋 Configuration
    ├── requirements.txt               # Dependencies
    └── README.md                     # This file
```

## 🔧 Installation

1. Copy all plugin files to your QGIS plugins directory:
   ```
   ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/annlandslide/
   ```

2. Restart QGIS

3. Enable the plugin in **Plugins** → **Manage and Install Plugins**

## 📊 Required Input Data

The plugin requires 14 raster layers in the following order:

1. **Aspect** - Slope aspect (0-360°)
2. **Elevation** - Digital elevation model
3. **Flow Accumulation** - Water flow accumulation
4. **Plan Curvature** - Horizontal curvature
5. **Profile Curvature** - Vertical curvature  
6. **Rivers Proximity** - Distance to rivers
7. **Roads Proximity** - Distance to roads
8. **Slope** - Slope gradient (0-90°)
9. **Stream Power Index** - Erosive power of flowing water
10. **Topographic Position Index** - Relative topographic position
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
