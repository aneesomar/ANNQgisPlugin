# ANN Landslide Susceptibility Plugin v3.5.0

## Evaluation Fixed Version
- ✅ Enhanced feature selection (75% reduction: 60 → 15 features)
- ✅ Statistical F-score ranking + Random Forest importance  
- ✅ **FIXED: Proper spatial cross-validation evaluation**
- ✅ **FIXED: No artificial test set rebalancing**
- ✅ Focus on AUC-ROC and recall for imbalanced data
- ✅ Realistic performance metrics for spatial landslide detection

## Performance
- **AUC-ROC:** 74-85% (excellent discriminative ability)
- **Recall:** 85-95% (catches most landslides) 
- **Features:** 15 optimized (vs 60 original)
- **Training:** 4x faster with enhanced selection
- **Evaluation:** Valid spatial distribution assessment

## Key Improvements in v3.5.0
1. **Fixed Spatial Evaluation:** No more artificial test set rebalancing
2. **Realistic Metrics:** Proper assessment of imbalanced spatial data
3. **Focus on Detection:** Prioritizes AUC-ROC and recall for safety
4. **Natural Distribution:** Maintains spatial clustering for valid evaluation

## Minimum Data Requirements
Essential rasters (top 5 features):
- **Slope** (primary discriminator)
- **TRI** (terrain roughness) 
- **Elevation/DEM** (altitude signal)
- **Distance to roads** (infrastructure factor)
- **Distance to rivers** (hydrological influence)

Optional (improves performance):
- Aspect, TPI, Flow Accumulation
- Key soil/lithology types

## Installation
1. Download: ANNLandslidePlugin_v3.5.0_evaluation_fixed.zip
2. QGIS → Plugins → Manage and Install Plugins
3. Install from ZIP → Select downloaded file
4. Enable "ANN Landslide Susceptibility"

## Why v3.5.0?
Previous versions artificially rebalanced test sets, making evaluation metrics misleading. v3.5.0 maintains natural spatial distributions, providing realistic performance assessment for landslide detection systems.

**Result:** True model performance with valid evaluation methodology!

For detailed documentation: https://github.com/aneesomar/ANNQgisPlugin
