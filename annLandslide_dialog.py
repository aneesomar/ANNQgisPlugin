# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AnnLandslideDialog - Simple Single Processor Version
 Basic version with proper progress tracking and single-threaded processing
 ***************************************************************************/
"""

import os
import sys

# Only import absolutely essential QGIS components
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QMessageBox

# Load the UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'annLandslide_dialog_base.ui'))

class AnnLandslideDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Simple constructor with basic functionality"""
        super(AnnLandslideDialog, self).__init__(parent)
        
        # Set up basic UI
        self.setupUi(self)
        
        # Show simple information about the plugin
        QMessageBox.information(
            self,
            "ANN Landslide Plugin - SIMPLE VERSION v2.0",
            f"✅ SIMPLE SINGLE-PROCESSOR VERSION LOADED\n\n"
            f"This version provides:\n"
            f"• Single-threaded processing (no system overload)\n"
            f"• Proper progress tracking (0-100%)\n"
            f"• Lower memory usage\n"
            f"• Maximum stability\n\n"
            f"This plugin requires machine learning libraries.\n"
            f"If you encounter issues, install dependencies:\n"
            f"pip install torch scikit-learn pandas numpy rasterio"
        )
        
        # Initialize with minimal functionality
        self.raster_combos = []
        self.expected_rasters = [
            'Aspect', 'Elevation', 'Flow Accumulation', 'Plan Curvature',
            'Profile Curvature', 'Rivers Proximity', 'Roads Proximity',
            'Slope', 'Stream Power Index', 'Topographic Position Index',
            'Terrain Ruggedness Index', 'Topographic Wetness Index',
            'Lithology', 'Soil'
        ]
        
        # Setup basic UI without heavy imports
        self.setup_basic_ui()
        self.connect_basic_signals()
    
    def setup_basic_ui(self):
        """Setup UI with minimal imports"""
        try:
            # Only import what we absolutely need
            from qgis.PyQt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
            from qgis.gui import QgsMapLayerComboBox
            from qgis.core import QgsMapLayerProxyModel
            
            # Get the scroll area content widget
            content_widget = self.scrollAreaWidgetContents
            layout = QVBoxLayout(content_widget)
            
            # Create combo boxes for each expected raster
            for i, raster_name in enumerate(self.expected_rasters):
                h_layout = QHBoxLayout()
                
                label = QLabel(f"{i+1}. {raster_name}:")
                label.setMinimumWidth(200)
                
                combo = QgsMapLayerComboBox()
                combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
                combo.setAllowEmptyLayer(True)
                combo.setCurrentIndex(-1)
                
                h_layout.addWidget(label)
                h_layout.addWidget(combo)
                layout.addLayout(h_layout)
                
                self.raster_combos.append(combo)
            
            content_widget.setLayout(layout)
            
            # Set default model path if exists
            plugin_dir = os.path.dirname(__file__)
            default_model = os.path.join(plugin_dir, 'landslide_model_advanced_complete.pth')
            if os.path.exists(default_model):
                self.lineEdit_model.setText(default_model)
                
        except Exception as e:
            QMessageBox.critical(self, "UI Setup Error", f"Error setting up UI: {str(e)}")
    
    def connect_basic_signals(self):
        """Connect signals safely"""
        try:
            self.pushButton_model.clicked.connect(self.browse_model)
            self.pushButton_output.clicked.connect(self.browse_output)
            self.button_box.accepted.connect(self.run_prediction_safe)
        except Exception as e:
            QMessageBox.warning(self, "Signal Error", f"Error connecting signals: {str(e)}")
    
    def browse_model(self):
        """Browse for model file"""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Trained Model", "", "PyTorch Models (*.pth);;All Files (*)")
            if file_path:
                self.lineEdit_model.setText(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Browse Error", f"Error browsing for model: {str(e)}")
    
    def browse_output(self):
        """Browse for output file"""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Susceptibility Map", "", "GeoTIFF Files (*.tif);;All Files (*)")
            if file_path:
                self.lineEdit_output.setText(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Browse Error", f"Error browsing for output: {str(e)}")
    
    def run_prediction_safe(self):
        """Simple single-threaded prediction with proper progress tracking"""
        try:
            # Validate basic inputs first
            model_path = self.lineEdit_model.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Input Error", "Please select a trained model file.")
                return
            
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Input Error", "Model file does not exist.")
                return
            
            output_path = self.lineEdit_output.text().strip()
            if not output_path:
                QMessageBox.warning(self, "Input Error", "Please specify an output path.")
                return
            
            # Check if any rasters are selected
            selected_count = sum(1 for combo in self.raster_combos if combo.currentLayer() is not None)
            if selected_count == 0:
                QMessageBox.warning(self, "Input Error", "Please select at least one raster layer.")
                return
            
            # Now run the simple prediction
            self.run_simple_prediction()
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error in prediction setup: {str(e)}")
    
    def run_simple_prediction(self):
        """Run prediction with simple single-threaded processing"""
        try:
            # Check for required modules
            missing_modules = []
            
            try:
                import torch
            except ImportError:
                missing_modules.append("torch")
            
            try:
                import sklearn
            except ImportError:
                missing_modules.append("scikit-learn")
            
            try:
                import pandas
            except ImportError:
                missing_modules.append("pandas")
            
            try:
                import numpy
            except ImportError:
                missing_modules.append("numpy")
            
            try:
                import rasterio
            except ImportError:
                missing_modules.append("rasterio")
            
            if missing_modules:
                QMessageBox.critical(
                    self,
                    "Missing Dependencies",
                    f"The following required packages are missing:\n\n{', '.join(missing_modules)}\n\n"
                    f"Please install them using:\n"
                    f"pip install {' '.join(missing_modules)}\n\n"
                    f"Then restart QGIS and try again."
                )
                return
            
            # Import the simple safe predictor
            try:
                from .landslide_model_simple_safe import LandslideSusceptibilityPredictor
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Model Import Error", 
                    f"Failed to import prediction model:\n\n{str(e)}\n\n"
                    f"Please check your installation and try again."
                )
                return
            
            # Run the actual prediction
            self.execute_prediction()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Prediction Error",
                f"Error in prediction: {str(e)}\n\n"
                f"Please check your inputs and try again."
            )
    
    def execute_prediction(self):
        """Execute the prediction with proper progress tracking"""
        try:
            # Import required modules
            from qgis.PyQt.QtWidgets import QProgressDialog
            from qgis.PyQt.QtCore import Qt
            from qgis.core import QgsProject, QgsRasterLayer
            from .landslide_model_simple_safe import LandslideSusceptibilityPredictor
            
            # Get inputs
            model_path = self.lineEdit_model.text().strip()
            output_path = self.lineEdit_output.text().strip()
            threshold = self.doubleSpinBox_threshold.value()
            
            # Get raster paths
            raster_paths = []
            for combo in self.raster_combos:
                layer = combo.currentLayer()
                if layer:
                    source = layer.source()
                    if '|' in source:
                        source = source.split('|')[0]
                    raster_paths.append(source)
            
            if len(raster_paths) != len(self.expected_rasters):
                QMessageBox.warning(
                    self, 
                    "Input Error", 
                    f"Please select all {len(self.expected_rasters)} required raster layers."
                )
                return
            
            # Create progress dialog
            progress = QProgressDialog("Initializing prediction...", "Cancel", 0, 100, self)
            progress.setWindowTitle("ANN Landslide Susceptibility")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Initialize predictor
            predictor = LandslideSusceptibilityPredictor()
            
            # Load model
            progress.setLabelText("Loading trained model...")
            progress.setValue(10)
            QtWidgets.QApplication.processEvents()
            
            predictor.load_model(model_path)
            predictor.threshold = threshold
            
            # Setup scaler
            progress.setLabelText("Setting up data scaler...")
            progress.setValue(20)
            QtWidgets.QApplication.processEvents()
            
            predictor.setup_scaler()
            
            # Process rasters
            progress.setLabelText("Processing rasters...")
            progress.setValue(30)
            QtWidgets.QApplication.processEvents()
            
            if progress.wasCanceled():
                return
            
            # Define progress callback
            def progress_callback(percent, message):
                if progress.wasCanceled():
                    raise Exception("Processing cancelled by user")
                
                # Map the progress from 30-95%
                mapped_progress = 30 + int((percent / 100.0) * 65)
                progress.setValue(mapped_progress)
                progress.setLabelText(message)
                QtWidgets.QApplication.processEvents()
            
            # Process with simple method (confirmed single-threaded)
            predictor.process_rasters_simple(raster_paths, output_path, progress_callback)
            
            # Finalize
            progress.setLabelText("Finalizing output...")
            progress.setValue(95)
            QtWidgets.QApplication.processEvents()
            
            progress.setValue(100)
            progress.close()
            
            # Success message
            reply = QMessageBox.question(
                self,
                "Processing Complete",
                f"Landslide susceptibility map created successfully!\n\n"
                f"Output: {output_path}\n\n"
                f"Would you like to add the result to the map?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                layer = QgsRasterLayer(output_path, "Landslide Susceptibility")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                else:
                    QMessageBox.warning(self, "Warning", "Could not add the result layer to the map.")
            
            self.accept()
            
        except Exception as e:
            if "cancelled" in str(e).lower():
                QMessageBox.information(self, "Cancelled", "Processing was cancelled by user.")
            else:
                QMessageBox.critical(
                    self,
                    "Prediction Error",
                    f"Error during prediction:\n\n{str(e)}\n\n"
                    f"Please check your inputs and try again."
                )
