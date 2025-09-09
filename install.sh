#!/bin/bash
# ANN Landslide Plugin Installation Script
# Installs the plugin to QGIS plugins directory

echo "🚀 Installing ANN Landslide QGIS Plugin..."

# Define plugin directory
PLUGIN_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins/annlandslide"

# Create plugin directory
echo "📁 Creating plugin directory..."
mkdir -p "$PLUGIN_DIR"

# Copy core plugin files
echo "📄 Copying plugin files..."
cp __init__.py "$PLUGIN_DIR/"
cp annLandslide.py "$PLUGIN_DIR/"
cp annLandslide_dialog.py "$PLUGIN_DIR/"
cp annLandslide_dialog_base.ui "$PLUGIN_DIR/"
cp landslide_model_simple_safe.py "$PLUGIN_DIR/"
cp metadata.txt "$PLUGIN_DIR/"
cp icon.png "$PLUGIN_DIR/"

# Copy internationalization files
if [ -d "i18n" ]; then
    echo "🌍 Copying translation files..."
    cp -r i18n "$PLUGIN_DIR/"
fi

echo "✅ Installation complete!"
echo "📋 Next steps:"
echo "   1. Restart QGIS"
echo "   2. Go to Plugins → Manage and Install Plugins"
echo "   3. Find 'ANN Landslide Susceptibility - Safe Version'"
echo "   4. Enable the plugin"
echo ""
echo "🎯 Plugin installed at: $PLUGIN_DIR"
