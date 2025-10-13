# ANN Landslide Susceptibility Plugin v3.4.0

## Enhanced Feature Selection
- Statistical F-score ranking
- 75% feature reduction (60 → 15 features)  
- 83.7% AUC-ROC performance
- Professional-grade accuracy

## Installation
1. Download plugin zip file
2. QGIS → Plugins → Manage and Install Plugins
3. Install from ZIP → Select downloaded file
4. Enable "ANN Landslide Susceptibility"

## Requirements
Minimum essential rasters:
- Slope (primary)
- Elevation/DEM 
- TRI (terrain roughness)
- Distance to roads
- Distance to rivers

## Performance
- 83.7% AUC-ROC (professional grade)
- 4x faster training
- 75% less memory usage
- Validated on 486 historical landslides

## Usage
1. Load required raster layers in QGIS
2. Click ANN Landslide plugin icon
3. Select training data points
4. Choose enhanced feature selection (default)
5. Train model and generate susceptibility map

For detailed documentation, visit: https://github.com/aneesomar/ANNQgisPlugin
