# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AnnLandslideDialog - Minimal Safe Version
 This version imports NO heavy dependencies at module level
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
        """Minimal safe constructor"""
        super(AnnLandslideDialog, self).__init__(parent)
        
        # Set up basic UI
        self.setupUi(self)
        
        # Show immediate warning about ML dependencies with performance info
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        QMessageBox.information(
            self,
            "ANN Landslide Plugin - High Performance Mode",
            f"🚀 HIGH PERFORMANCE VERSION 🚀\n\n"
            f"• Parallel processing enabled ({cpu_count} CPU cores detected)\n"
            f"• Optimized memory management\n"
            f"• Fast I/O operations\n\n"
            f"This plugin requires machine learning libraries.\n"
            f"If you encounter issues, install dependencies:\n"
            f"pip3 install --user torch scikit-learn pandas numpy rasterio\n\n"
            f"Processing will be significantly faster than before!"
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
        """Ultra-safe prediction runner"""
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
            
            # Now try to run the actual prediction with full error isolation
            self.run_prediction_isolated()
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error in prediction setup: {str(e)}")
    
    def run_prediction_isolated(self):
        """Run prediction in completely isolated environment"""
        try:
            # Test if we can import required modules
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
                    f"pip3 install --user {' '.join(missing_modules)}\n\n"
                    f"Then restart QGIS and try again."
                )
                return
            
            # If all modules are available, try to import the fast model
            try:
                from .landslide_model_fast import FastLandslideSusceptibilityPredictor
            except Exception as e:
                # Fallback to regular model
                try:
                    from .landslide_model import LandslideSusceptibilityPredictor
                    FastLandslideSusceptibilityPredictor = LandslideSusceptibilityPredictor
                except Exception as e2:
                    QMessageBox.critical(
                        self,
                        "Model Import Error", 
                        f"Failed to import prediction model:\n\n{str(e)}\n\n"
                        f"Fallback also failed: {str(e2)}\n\n"
                        f"Please check your installation and try again."
                    )
                    return
            
            # If we get here, try to run the actual prediction
            self.run_actual_prediction()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Isolated Prediction Error",
                f"Error in isolated prediction: {str(e)}\n\n"
                f"Please check the TROUBLESHOOTING.md file for solutions."
            )
    
    def run_actual_prediction(self):
        """Run the actual prediction with all safety checks passed"""
        try:
            # Import all required modules here (after safety checks)
            from qgis.PyQt.QtWidgets import QProgressDialog
            from qgis.PyQt.QtCore import Qt
            from qgis.core import QgsProject, QgsRasterLayer
            from .landslide_model_fast import FastLandslideSusceptibilityPredictor
            
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
            
            # Show performance options dialog
            import multiprocessing as mp
            max_workers = mp.cpu_count()
            
            from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton, QDialogButtonBox
            
            # Create performance options dialog
            perf_dialog = QDialog(self)
            perf_dialog.setWindowTitle("Performance Settings")
            perf_dialog.setModal(True)
            
            layout = QVBoxLayout(perf_dialog)
            
            # Parallel processing option
            parallel_check = QCheckBox("Enable parallel processing (recommended)")
            parallel_check.setChecked(True)
            layout.addWidget(parallel_check)
            
            # Number of workers
            worker_layout = QHBoxLayout()
            worker_layout.addWidget(QLabel("Number of CPU cores to use:"))
            worker_spin = QSpinBox()
            worker_spin.setMinimum(1)
            worker_spin.setMaximum(max_workers)
            worker_spin.setValue(max(1, max_workers - 1))  # Leave one core free
            worker_layout.addWidget(worker_spin)
            worker_layout.addWidget(QLabel(f"(Max: {max_workers})"))
            layout.addLayout(worker_layout)
            
            # Chunk size option
            chunk_layout = QHBoxLayout()
            chunk_layout.addWidget(QLabel("Chunk size (pixels per batch):"))
            chunk_spin = QSpinBox()
            chunk_spin.setMinimum(10000)
            chunk_spin.setMaximum(200000)
            chunk_spin.setValue(50000)
            chunk_spin.setSingleStep(10000)
            chunk_layout.addWidget(chunk_spin)
            layout.addLayout(chunk_layout)
            
            # Info label
            info_label = QLabel("Higher chunk size = more memory usage but potentially faster processing.\nParallel processing will significantly speed up large datasets.")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
            
            # Dialog buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(perf_dialog.accept)
            buttons.rejected.connect(perf_dialog.reject)
            layout.addWidget(buttons)
            
            # Show performance dialog
            if perf_dialog.exec_() != QDialog.Accepted:
                return
            
            # Get settings
            use_parallel = parallel_check.isChecked()
            num_workers = worker_spin.value() if use_parallel else 1
            chunk_size = chunk_spin.value()
            
            # Show progress dialog
            progress = QProgressDialog("Initializing high-performance prediction...", "Cancel", 0, 100, self)
            progress.setWindowTitle("ANN Landslide Susceptibility - Turbo Mode")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Initialize and run fast predictor
            predictor = FastLandslideSusceptibilityPredictor(
                use_parallel=use_parallel,
                num_workers=num_workers
            )
            
            progress.setLabelText("Loading model...")
            progress.setValue(10)
            predictor.load_model(model_path)
            predictor.threshold = threshold
            
            progress.setLabelText("Setting up scaler...")
            progress.setValue(20)
            predictor.setup_scaler()
            
            progress.setLabelText("Starting high-performance processing...")
            progress.setValue(30)
            
            # Process rasters with performance monitoring
            import time
            start_time = time.time()
            
            def progress_callback(msg):
                progress.setLabelText(msg)
                QtWidgets.QApplication.processEvents()
                
                # Cancel check
                if progress.wasCanceled():
                    raise Exception("Processing cancelled by user")
            
            # Use parallel processing method
            predictor.process_rasters(raster_paths, output_path, chunk_size, progress_callback)
            
            total_time = time.time() - start_time
            
            progress.setValue(100)
            progress.close()
            
            # Success message with performance info
            mode_text = f"Parallel ({num_workers} cores)" if use_parallel else "Sequential"
            reply = QMessageBox.question(
                self,
                "🚀 High Performance Processing Complete! 🚀",
                f"Susceptibility map created successfully!\n\n"
                f"⚡ Processing mode: {mode_text}\n"
                f"⏱️ Total time: {total_time:.1f} seconds\n"
                f"📊 Chunk size: {chunk_size:,} pixels\n\n"
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
            QMessageBox.critical(
                self,
                "Prediction Execution Error",
                f"Error during prediction execution:\n\n{str(e)}\n\n"
                f"Please check your inputs and try again."
            )
