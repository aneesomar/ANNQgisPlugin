#!/bin/bash

# ANN Landslide QGIS Plugin Packaging Script
# Creates a zip file ready for QGIS plugin installation

echo "🚀 Creating QGIS Plugin Package..."

# Plugin details
PLUGIN_NAME="annlandslide_plugin"
PLUGIN_DISPLAY_NAME="ANN Landslide"
VERSION="3.1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PACKAGE_NAME="${PLUGIN_NAME}_v${VERSION}_${TIMESTAMP}"

# Create temporary packaging directory
TEMP_DIR="/tmp/${PACKAGE_NAME}"
echo "📁 Setting up package directory: $TEMP_DIR"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Core plugin files that must be included
CORE_FILES=(
    "__init__.py"
    "annLandslide.py"
    "annLandslide_dialog.py" 
    "comprehensive_training_dialog.py"
    "ann_training_module.py"
    "csv_only_training.py"
    "simple_training_module.py"
    "landslide_model_simple_safe.py"
    "raster_data_extractor.py"
    "icon.png"
)

# UI files
UI_FILES=(
    "annLandslide_dialog_base.ui"
    "model_training_dialog_base.ui"
)

# Copy core plugin files
echo "📋 Copying core plugin files..."
for file in "${CORE_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$TEMP_DIR/"
        echo "  ✅ $file"
    else
        echo "  ⚠️  $file (not found - optional)"
    fi
done

# Copy UI files
echo "📋 Copying UI files..."
for file in "${UI_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$TEMP_DIR/"
        echo "  ✅ $file"
    else
        echo "  ⚠️  $file (not found)"
    fi
done

# Copy internationalization files if they exist
if [ -d "i18n" ]; then
    echo "📋 Copying internationalization files..."
    cp -r "i18n" "$TEMP_DIR/"
    echo "  ✅ i18n/ directory"
fi

# Create metadata.txt
echo "📋 Creating metadata.txt..."
cat > "$TEMP_DIR/metadata.txt" << EOF
[general]
name=$PLUGIN_DISPLAY_NAME
qgisMinimumVersion=3.0
description=Advanced Neural Network plugin for landslide susceptibility mapping and model training
version=$VERSION
author=Anees Omar
email=aneesomar@example.com

about=This plugin provides comprehensive landslide susceptibility analysis using Artificial Neural Networks (ANN). Features include:
    - Automated raster sampling from vector points
    - Advanced neural network training with multiple fallback methods
    - CPU-only processing for maximum compatibility
    - Sample data generation for testing
    - Complete landslide susceptibility mapping workflow

tracker=https://github.com/aneesomar/ANNQgisPlugin/issues
repository=https://github.com/aneesomar/ANNQgisPlugin

tags=landslide,neural network,machine learning,susceptibility,mapping,training,ANN

homepage=https://github.com/aneesomar/ANNQgisPlugin
category=Analysis
icon=icon.png
experimental=False
deprecated=False
server=False
EOF

echo "  ✅ metadata.txt created"

# Create requirements.txt for easy dependency installation
echo "📋 Creating requirements.txt..."
cat > "$TEMP_DIR/requirements.txt" << EOF
torch
scikit-learn
pandas
numpy
EOF

echo "  ✅ requirements.txt created"

# Create installation guide
echo "📋 Creating INSTALL.md..."
cat > "$TEMP_DIR/INSTALL.md" << EOF
# ANN Landslide Plugin Installation Guide

## 1. Install Plugin
1. Open QGIS
2. Go to **Plugins** → **Manage and Install Plugins**
3. Click **Install from ZIP**
4. Select this ZIP file
5. Enable the plugin

## 2. Install Dependencies
In QGIS Python Console, run:
\`\`\`python
import subprocess, sys
packages = ['torch', 'scikit-learn', 'pandas', 'numpy']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
\`\`\`

## 3. Usage
- **Train Model**: Plugins → ANN Landslide → Train New Model
- **Run Prediction**: Plugins → ANN Landslide → Run Landslide Susceptibility

## 4. Features
- ✅ Automated raster sampling and training
- ✅ Multiple training fallback methods
- ✅ CPU-only processing (no CUDA issues)
- ✅ Sample data generation for testing

## 5. Troubleshooting
- If training fails, it automatically falls back to sample data generation
- All processing uses CPU to avoid CUDA compatibility issues
- Check QGIS Python Console for detailed error messages
EOF

echo "  ✅ INSTALL.md created"

# Create the ZIP package
ZIP_FILE="/tmp/${PACKAGE_NAME}.zip"
echo "📦 Creating ZIP package..."
cd /tmp
zip -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}/"

# Copy to project directory
PROJECT_ZIP="./annlandslide_qgis_ready.zip"
cp "$ZIP_FILE" "$PROJECT_ZIP"

# Display package contents
echo ""
echo "📄 Package contents:"
unzip -l "$ZIP_FILE" | head -20

# Display summary
echo ""
echo "✅ QGIS Plugin Package Created Successfully!"
echo "📦 Package: $PROJECT_ZIP"
echo "📁 Size: $(du -h "$PROJECT_ZIP" | cut -f1)"
echo ""
echo "🚀 Installation Instructions:"
echo "1. Open QGIS"
echo "2. Plugins → Manage and Install Plugins"
echo "3. Install from ZIP → Select '$PROJECT_ZIP'"
echo "4. Install dependencies (see INSTALL.md in package)"
echo ""
echo "📋 Package includes:"
echo "   • Core plugin files (Python)"
echo "   • UI interface files"
echo "   • Plugin metadata and configuration"
echo "   • Installation guide and requirements"
echo "   • Ready for QGIS installation"

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Ready to install in QGIS!"