# Installation Guide - ANN Landslide Plugin v3.4.0

## Quick Installation

### Method 1: QGIS Plugin Manager (Recommended)
1. Open QGIS
2. Go to **Plugins** → **Manage and Install Plugins**
3. Click **Install from ZIP**
4. Select `ANNLandslidePlugin_v3.4.0_feature_selection.zip`
5. Click **Install Plugin**
6. Enable **ANN Landslide Susceptibility** in plugin list

### Method 2: Manual Installation
1. Extract `ANNLandslidePlugin_v3.4.0_feature_selection.zip`
2. Copy `ANNLandslidePlugin` folder to:
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Mac**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable plugin in Plugin Manager

## System Requirements

### Software:
- QGIS 3.16+ (recommended: 3.28+)
- Python 3.8+ with PyTorch support

### Python Dependencies:
- torch >= 1.9.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- geopandas >= 0.10.0

### Hardware (Recommended):
- RAM: 8GB+ (16GB for large datasets)
- CPU: Multi-core processor
- GPU: Optional (CUDA support for large models)
- Storage: 2GB+ free space

## First-Time Setup

### 1. Verify Installation
- Look for ANN Landslide icon in QGIS toolbar
- Check Plugins menu for "ANN Landslide Susceptibility"

### 2. Test with Sample Data
- Use provided sample rasters for testing
- Follow tutorial workflow in documentation

### 3. Configure Python Environment
- Ensure PyTorch is installed in QGIS Python environment
- Test model training with small dataset

## Troubleshooting

### Common Issues:

**Plugin not appearing:**
- Restart QGIS after installation
- Check if plugin is enabled in Plugin Manager
- Verify correct installation directory

**Import errors:**
- Install missing Python packages
- Check QGIS Python console for error details
- Update to compatible package versions

**Training failures:**
- Verify input data alignment and CRS
- Check landslide point data format
- Ensure sufficient memory available

### Getting Help:
1. Check README.md for detailed usage instructions
2. Review CHANGELOG.md for version-specific information
3. Submit issues to GitHub repository
4. Consult QGIS documentation for general troubleshooting

## Next Steps

1. **Prepare Data**: Collect and align required raster datasets
2. **Follow Tutorial**: Use step-by-step workflow guide
3. **Train Model**: Start with small dataset for testing
4. **Validate Results**: Compare against known landslide inventory
5. **Generate Maps**: Create susceptibility maps for your study area

---
**Happy Mapping! 🗺️**