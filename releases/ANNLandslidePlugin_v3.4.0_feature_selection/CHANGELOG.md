# Changelog - ANN Landslide Susceptibility Plugin

## Version 3.4.0 (October 13, 2025) - Feature Selection Enhanced

### 🚀 Major Features:
- **Enhanced Feature Selection**: Statistical F-score and RF importance-based selection
- **Quality Filtering**: Automatic removal of 37+ weak categorical features
- **75% Feature Reduction**: Optimized from 60+ to 15 discriminative features
- **Improved Performance**: 83.7% AUC-ROC with streamlined feature set

### 🔧 Technical Improvements:
- Statistical feature ranking (F-test, Random Forest importance)
- Quality-based filtering (low-variance, weak categorical removal)
- Optimized training pipeline with spatial cross-validation
- Enhanced model architecture with Focal Loss
- Professional-grade validation against historical landslides

### 📊 Performance Gains:
- AUC-ROC: 83.7% (excellent discriminative ability)
- Feature efficiency: 76% reduction (62 → 15 features)
- Training speed: 4x faster with fewer features
- Model robustness: Reduced overfitting through noise removal

### 🎯 Key Features Identified:
1. Slope (F-score: 4604) - Primary discriminator
2. Elevation (F-score: 1422) - Strong topographic signal
3. TRI, Road/River Proximity - Key secondary features

### ✅ Validation Results:
- 80.5% historical landslide capture rate
- Statistically significant discrimination (p < 0.001)
- Balanced spatial cross-validation
- Professional mapping standards achieved

## Version 3.3.0 (Previous)
- Enhanced training parameters
- PR-AUC metrics integration
- Improved spatial cross-validation

## Version 3.2.0 (Previous)  
- Advanced threshold optimization
- Spatial cross-validation fixes
- Performance improvements

## Version 3.1.0 (Previous)
- Initial enhanced performance model
- Basic feature selection
- QGIS integration improvements