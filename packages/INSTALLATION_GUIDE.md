# 📦 Quick Installation Guide - ZIP Package

## 🚀 Easy QGIS Plugin Installation

### Step-by-Step Instructions:

1. **📁 Locate the ZIP Package**
   - File: `packages/annlandslide_v2.1.zip`
   - Size: ~14 KB
   - Contains all plugin files

2. **🎯 Install in QGIS**
   - Open QGIS
   - Go to **Plugins** → **Manage and Install Plugins**
   - Click **"Install from ZIP"** tab
   - Click **"..."** button to browse
   - Select `annlandslide_v2.1.zip`
   - Click **"Install Plugin"**

3. **✅ Enable Plugin**
   - Go to **"Installed"** tab
   - Find "ANN Landslide Susceptibility - Safe Version"
   - Check the checkbox to enable
   - Click **"Close"**

4. **🎉 Ready to Use**
   - Look for the plugin icon in QGIS toolbar
   - Click to open the landslide prediction dialog
   - Load your 14 raster inputs and process!

## 📋 What's Included in the ZIP:

- ✅ Core plugin files (`__init__.py`, `annLandslide.py`, etc.)
- ✅ User interface components (`*.ui` files)
- ✅ Safe processing model (`landslide_model_simple_safe.py`)
- ✅ Plugin metadata and icon
- ✅ Translation files (internationalization)
- ✅ Dependencies list

## 🔍 Verification:

After installation, you should see:
- Plugin name: **"ANN Landslide Susceptibility - Safe Version"**
- Version: **2.1**
- Status: **Enabled**
- Icon: Available in QGIS toolbar

## 🆘 Troubleshooting:

- **Plugin not appearing?** → Check if it's enabled in the Installed tab
- **Installation failed?** → Try manual installation with `install.sh`
- **Missing dependencies?** → Install: `pip install torch scikit-learn rasterio`

---

**ZIP Package**: `annlandslide_v2.1.zip`  
**Version**: 2.1 (Safe Version)  
**Status**: ✅ Ready for distribution
