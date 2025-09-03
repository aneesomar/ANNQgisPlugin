# Quick Start Guide - ANN Landslide Susceptibility Plugin

## Overview
This QGIS plugin generates landslide susceptibility maps using a trained Artificial Neural Network (ANN) model.

## Quick Setup

### 1. Install Dependencies
```bash
# Run the installation script
./install_dependencies.sh

# OR install manually
pip3 install --user torch scikit-learn pandas rasterio numpy matplotlib seaborn scipy
```

### 2. Install Plugin
1. Copy the entire plugin folder to your QGIS plugins directory:
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/annlandslide/`
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\annlandslide\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/annlandslide/`

2. Restart QGIS

3. Enable the plugin: **Plugins** → **Manage and Install Plugins** → **Installed** → Check "ANN Landslide Susceptibility"

### 3. Test Installation
```bash
# Run the test script
python3 test_plugin.py
```

## Usage

### 1. Prepare Input Data
You need 14 raster layers (all with same extent, resolution, and CRS):

1. **Aspect** - Terrain aspect in degrees
2. **Elevation** - Digital elevation model
3. **Flow Accumulation** - Water flow accumulation
4. **Plan Curvature** - Plan curvature of terrain
5. **Profile Curvature** - Profile curvature of terrain  
6. **Rivers Proximity** - Distance to nearest river/stream
7. **Roads Proximity** - Distance to nearest road
8. **Slope** - Terrain slope (degrees or percentage)
9. **Stream Power Index (SPI)** - Stream power index
10. **Topographic Position Index (TPI)** - Topographic position index
11. **Terrain Ruggedness Index (TRI)** - Terrain ruggedness index
12. **Topographic Wetness Index (TWI)** - Topographic wetness index
13. **Lithology** - Lithological units (integer codes 1-14)
14. **Soil** - Soil types (integer codes 1-11)

### 2. Load Data in QGIS
- Add all 14 raster layers to your QGIS project
- Ensure all layers have the same spatial reference system
- Check that all layers cover the same geographic extent

### 3. Run the Plugin
1. Open: **Plugins** → **ANN Landslide Susceptibility**
2. **Select Trained Model**: Browse to your `.pth` model file
3. **Set Output Path**: Choose where to save the result (`.tif` file)
4. **Select Input Rasters**: For each of the 14 layers, select the corresponding raster from the dropdown
5. **Adjust Threshold** (optional): Default is 0.5
6. **Click OK** to start processing

### 4. View Results
- The plugin will process the data in chunks (memory efficient)
- Progress will be shown during processing
- Result will be automatically added to the map (optional)
- Output values range from 0 (low susceptibility) to 1 (high susceptibility)

## Tips for Best Results

### Data Preparation
- **Align all rasters**: Use QGIS "Align Rasters" tool or GDAL to ensure consistent grid
- **Check data quality**: Remove or interpolate any missing values
- **Consistent units**: Ensure distance measurements use the same units
- **Proper classification**: Lithology and soil should use integer codes

### Performance
- **Large datasets**: The plugin processes data in chunks to avoid memory issues
- **Processing time**: Depends on raster size and computer specs
- **Memory usage**: Close other applications for large datasets

### Common Issues
- **"Raster dimensions don't match"**: Align all input rasters first
- **"Model loading error"**: Check model file compatibility
- **"Missing dependencies"**: Run the installation script again

## Model Information

The plugin includes a pre-trained model with:
- **Input features**: 60 selected features (from 37 derived features)
- **Architecture**: Advanced ANN with attention mechanisms and residual blocks
- **Training threshold**: ~0.45 (optimized for best performance)

## Support

- **Test your setup**: Run `python3 test_plugin.py`
- **Check dependencies**: Run `./install_dependencies.sh`
- **Issues**: Create issue on GitHub or email aneesomar.ao@gmail.com

## File Structure
```
annlandslide/
├── annLandslide.py                 # Main plugin class
├── annLandslide_dialog.py          # Dialog interface
├── annLandslide_dialog_base.ui     # UI layout
├── landslide_model.py              # ANN model implementation
├── landslide_model_advanced_complete.pth  # Trained model
├── requirements.txt                # Python dependencies
├── install_dependencies.sh         # Installation script
├── test_plugin.py                 # Test script
├── README.md                      # Full documentation
├── QUICKSTART.md                  # This file
└── metadata.txt                   # QGIS plugin metadata
```
