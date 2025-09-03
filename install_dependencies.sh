#!/bin/bash

# ANN Landslide Susceptibility Plugin - Dependency Installation Script
# This script helps install the required Python packages for QGIS

echo "ANN Landslide Susceptibility Plugin - Dependency Installer"
echo "========================================================="
echo ""

# Function to detect QGIS Python path
detect_qgis_python() {
    echo "Detecting QGIS Python environment..."
    
    # Common QGIS Python paths
    QGIS_PYTHON_PATHS=(
        "/usr/bin/python3"  # System Python (used by QGIS on many Linux systems)
        "/Applications/QGIS.app/Contents/MacOS/bin/python3"  # macOS QGIS
        "$(which python3)"  # User's default Python3
    )
    
    for python_path in "${QGIS_PYTHON_PATHS[@]}"; do
        if [ -x "$python_path" ]; then
            echo "Found Python at: $python_path"
            PYTHON_CMD="$python_path"
            return 0
        fi
    done
    
    echo "Could not automatically detect QGIS Python. Using system default."
    PYTHON_CMD="python3"
}

# Function to install packages
install_packages() {
    echo ""
    echo "Installing required packages..."
    echo "Note: This may take several minutes depending on your internet connection."
    echo ""
    
    # Packages to install
    packages=(
        "torch>=1.9.0"
        "torchvision>=0.10.0" 
        "scikit-learn>=1.0.0"
        "pandas>=1.3.0"
        "numpy>=1.21.0"
        "rasterio>=1.2.0"
        "matplotlib>=3.4.0"
        "seaborn>=0.11.0"
        "scipy>=1.7.0"
    )
    
    for package in "${packages[@]}"; do
        echo "Installing $package..."
        $PYTHON_CMD -m pip install "$package" --user
        
        if [ $? -eq 0 ]; then
            echo "✓ $package installed successfully"
        else
            echo "✗ Failed to install $package"
            failed_packages+=("$package")
        fi
    done
}

# Function to verify installation
verify_installation() {
    echo ""
    echo "Verifying installation..."
    echo ""
    
    # Test imports
    test_imports=(
        "torch"
        "sklearn"
        "pandas" 
        "numpy"
        "rasterio"
        "matplotlib"
        "seaborn"
        "scipy"
    )
    
    failed_imports=()
    
    for module in "${test_imports[@]}"; do
        if $PYTHON_CMD -c "import $module" 2>/dev/null; then
            echo "✓ $module can be imported"
        else
            echo "✗ $module import failed"
            failed_imports+=("$module")
        fi
    done
    
    if [ ${#failed_imports[@]} -eq 0 ]; then
        echo ""
        echo "🎉 All packages installed successfully!"
        echo "The ANN Landslide Susceptibility plugin should now work properly."
    else
        echo ""
        echo "⚠️  Some packages failed to install or import:"
        printf '%s\n' "${failed_imports[@]}"
        echo ""
        echo "Try installing them manually using:"
        echo "$PYTHON_CMD -m pip install --user <package_name>"
    fi
}

# Main execution
main() {
    detect_qgis_python
    install_packages
    verify_installation
    
    echo ""
    echo "Installation script completed."
    echo "If you encounter any issues, please check the plugin documentation."
}

# Run main function
main
