#!/bin/bash

# Package the ANN Landslide Susceptibility Plugin for distribution
# This script creates a zip file ready for QGIS plugin installation

PLUGIN_NAME="annlandslide"
VERSION="1.0"
PACKAGE_NAME="${PLUGIN_NAME}-${VERSION}"

echo "Packaging ANN Landslide Susceptibility Plugin"
echo "=============================================="

# Create temporary directory
TEMP_DIR="/tmp/${PACKAGE_NAME}"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "Copying plugin files..."

# Core plugin files
cp annLandslide.py "$TEMP_DIR/"
cp annLandslide_dialog.py "$TEMP_DIR/"
cp annLandslide_dialog_base.ui "$TEMP_DIR/"
cp landslide_model.py "$TEMP_DIR/"
cp resources.py "$TEMP_DIR/"
cp __init__.py "$TEMP_DIR/"
cp metadata.txt "$TEMP_DIR/"
cp icon.png "$TEMP_DIR/"

# Model file (if exists)
if [ -f "landslide_model_advanced_complete.pth" ]; then
    echo "Including trained model..."
    cp landslide_model_advanced_complete.pth "$TEMP_DIR/"
fi

# Documentation
cp README.md "$TEMP_DIR/"
cp QUICKSTART.md "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"

# Scripts
cp install_dependencies.sh "$TEMP_DIR/"
cp test_plugin.py "$TEMP_DIR/"
chmod +x "$TEMP_DIR/install_dependencies.sh"
chmod +x "$TEMP_DIR/test_plugin.py"

# Create the zip package
echo "Creating zip package..."
cd /tmp
zip -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}/" -x "*.pyc" "*__pycache__*"

# Move to current directory
mv "${PACKAGE_NAME}.zip" "$(pwd)/"

echo ""
echo "✓ Package created: ${PACKAGE_NAME}.zip"
echo ""
echo "Installation instructions:"
echo "1. In QGIS, go to Plugins → Manage and Install Plugins"
echo "2. Click 'Install from ZIP'"
echo "3. Select the ${PACKAGE_NAME}.zip file"
echo "4. Install required dependencies using install_dependencies.sh"
echo ""
echo "Or manually install:"
echo "1. Extract the zip to your QGIS plugins directory"
echo "2. Restart QGIS"
echo "3. Enable the plugin in Plugin Manager"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Package ready for distribution!"
