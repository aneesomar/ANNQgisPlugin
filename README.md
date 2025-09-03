# ANN Landslide Susceptibility QGIS Plugin

A QGIS plugin for generating landslide susceptibility maps using trained Artificial Neural Network (ANN) models.

## Features

- **Advanced ANN Architecture**: Uses a sophisticated neural network with attention mechanisms and residual blocks
- **Raster Processing**: Handles multiple input raster layers with automatic alignment and processing
- **Memory Efficient**: Processes large datasets in chunks to avoid memory issues
- **QGIS Integration**: Seamlessly integrates with QGIS layer management and visualization
- **Customizable Thresholds**: Allows users to adjust prediction thresholds
- **Progress Tracking**: Real-time progress updates during processing

## Requirements

### Software Requirements
- QGIS 3.0 or higher
- Python 3.7 or higher

### Python Dependencies
Install the following packages in your QGIS Python environment:

```bash
pip install torch torchvision scikit-learn pandas numpy rasterio matplotlib seaborn scipy
```

Or use the provided requirements.txt:
```bash
pip install -r requirements.txt
```

## Installation

1. **Download the Plugin**: Clone or download this repository
2. **Install Dependencies**: Install the required Python packages (see Requirements above)
3. **Copy to QGIS Plugins Directory**: 
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - Windows: `C:\\Users\\[username]\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
4. **Enable the Plugin**: In QGIS, go to Plugins → Manage and Install Plugins → Installed, and enable "ANN Landslide Susceptibility"

## Usage

### Input Data Requirements

The plugin expects 14 raster layers in the following order:

1. **Aspect** - Terrain aspect (degrees)
2. **Elevation** - Digital elevation model
3. **Flow Accumulation** - Water flow accumulation
4. **Plan Curvature** - Plan curvature of terrain
5. **Profile Curvature** - Profile curvature of terrain
6. **Rivers Proximity** - Distance to nearest river
7. **Roads Proximity** - Distance to nearest road
8. **Slope** - Terrain slope (degrees or percentage)
9. **Stream Power Index (SPI)** - Stream power index
10. **Topographic Position Index (TPI)** - Topographic position index
11. **Terrain Ruggedness Index (TRI)** - Terrain ruggedness index
12. **Topographic Wetness Index (TWI)** - Topographic wetness index
13. **Lithology** - Lithological units (classified raster)
14. **Soil** - Soil types (classified raster)

### Steps to Generate Susceptibility Map

1. **Open the Plugin**: In QGIS, go to Plugins → ANN Landslide Susceptibility
2. **Select Trained Model**: Browse and select your trained ANN model file (.pth)
3. **Set Output Path**: Choose where to save the susceptibility map (.tif)
4. **Select Input Rasters**: For each of the 14 required layers, select the corresponding raster layer from your QGIS project
5. **Set Threshold** (Optional): Adjust the prediction threshold (default: 0.5)
6. **Run Prediction**: Click OK to start the processing

The plugin will:
- Process rasters in memory-efficient chunks
- Apply the trained ANN model
- Generate a continuous susceptibility map (0-1 values)
- Optionally add the result to your QGIS map

## Model Architecture

The plugin uses an advanced ANN architecture featuring:

- **Attention Mechanisms**: For focusing on important features
- **Residual Blocks**: For better gradient flow and model performance
- **Batch Normalization**: For stable training and prediction
- **Dropout Layers**: For regularization and robustness
- **Deep Architecture**: Multiple hidden layers for complex pattern recognition

## Data Preprocessing

The plugin automatically handles:

- **Feature Scaling**: Continuous variables are normalized using MinMaxScaler
- **One-Hot Encoding**: Categorical variables (lithology, soil) are one-hot encoded
- **Feature Selection**: Uses the same features selected during model training
- **Missing Data**: Handles NaN values appropriately

## Output

The plugin generates:
- **Susceptibility Raster**: Continuous values from 0 (low susceptibility) to 1 (high susceptibility)
- **GeoTIFF Format**: Compatible with QGIS and other GIS software
- **Spatial Reference**: Preserves the CRS from input rasters

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required Python packages are installed in your QGIS Python environment
2. **Memory Errors**: Reduce chunk size for very large datasets
3. **Model Loading Errors**: Ensure the model file is compatible and not corrupted
4. **Raster Alignment**: All input rasters should have the same extent, resolution, and CRS

### Error Messages

- **"Missing required dependencies"**: Install the required Python packages
- **"Model file structure not recognized"**: Use a compatible model file
- **"Raster dimensions don't match"**: Ensure all input rasters are properly aligned

## Model Training

If you need to train your own model:

1. Prepare training data with the required features
2. Use the provided training scripts in the `/python` directory
3. Save the trained model with the complete architecture and metadata
4. Use the saved model (.pth file) with this plugin

## Support

For issues, questions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/aneesomar/ANNQgisPlugin/issues)
- **Email**: aneesomar.ao@gmail.com

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

## Citation

If you use this plugin in your research, please cite:
```
Omar, A. (2025). ANN Landslide Susceptibility QGIS Plugin. GitHub Repository: https://github.com/aneesomar/ANNQgisPlugin
```

## Version History

- **v1.0** (2025-09-03): Initial release with ANN model integration, raster processing, and QGIS integration
