# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ModelTrainingDialog
 Dialog for training ANN landslide susceptibility models
 ***************************************************************************/
"""

import os
import numpy as np
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import QThread, pyqtSignal, QTimer
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel
from qgis.gui import QgsMapLayerComboBox, QgsFileWidget
from qgis.core import (QgsMapLayerProxyModel, QgsVectorLayer, QgsRasterLayer, 
                       QgsProject, QgsCoordinateReferenceSystem, QgsPointXY,
                       QgsGeometry, QgsFeature, QgsFields, QgsField,
                       QgsVectorFileWriter, QgsSpatialIndex)
from qgis.PyQt.QtCore import QVariant
import random

# Load the UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'model_training_dialog_base.ui'))

class ModelTrainingWorker(QThread):
    """Worker thread for model training to prevent UI freezing"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, raster_paths, landslide_points_path, output_model_path, 
                 epochs=150, batch_size=64, test_split=0.2, generate_non_landslides=True):
        super().__init__()
        self.raster_paths = raster_paths
        self.landslide_points_path = landslide_points_path
        self.output_model_path = output_model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_split = test_split
        self.generate_non_landslides = generate_non_landslides
        
    def run(self):
        """Run the training process"""
        try:
            self.status.emit("Starting model training...")
            self.progress.emit(5)
            
            # Import training module
            from .ann_training_module import ANNTrainingModule
            
            trainer = ANNTrainingModule()
            
            # Step 1: Extract features from rasters
            self.status.emit("Extracting features from rasters...")
            self.progress.emit(10)
            
            feature_data = trainer.extract_features_from_rasters(
                self.raster_paths, 
                self.landslide_points_path,
                generate_non_landslides=self.generate_non_landslides,
                progress_callback=self.update_feature_progress
            )
            
            self.progress.emit(40)
            
            # Step 2: Prepare training data
            self.status.emit("Preparing training data...")
            X, y, selected_features, scaler = trainer.prepare_training_data(
                feature_data, 
                test_split=self.test_split
            )
            
            self.progress.emit(50)
            
            # Step 3: Train model
            self.status.emit("Training neural network...")
            model, training_info = trainer.train_model(
                X, y, 
                epochs=self.epochs,
                batch_size=self.batch_size,
                progress_callback=self.update_training_progress
            )
            
            self.progress.emit(90)
            
            # Step 4: Save model
            self.status.emit("Saving trained model...")
            trainer.save_model(
                model, scaler, selected_features, training_info, 
                self.output_model_path
            )
            
            self.progress.emit(100)
            self.status.emit("Model training completed successfully!")
            self.finished.emit(True, f"Model saved to: {self.output_model_path}")
            
        except Exception as e:
            self.finished.emit(False, f"Training failed: {str(e)}")
    
    def update_feature_progress(self, progress):
        """Update progress during feature extraction"""
        # Progress from 10-40 (30% range)
        adjusted_progress = 10 + int(progress * 0.3)
        self.progress.emit(adjusted_progress)
    
    def update_training_progress(self, epoch, total_epochs):
        """Update progress during training"""
        # Progress from 50-90 (40% range)
        epoch_progress = epoch / total_epochs
        adjusted_progress = 50 + int(epoch_progress * 40)
        self.progress.emit(adjusted_progress)
        self.status.emit(f"Training epoch {epoch}/{total_epochs}")

class ModelTrainingDialog(QtWidgets.QDialog, FORM_CLASS):
    """Dialog for training ANN models"""
    
    def __init__(self, parent=None):
        super(ModelTrainingDialog, self).__init__(parent)
        self.setupUi(self)
        
        # Initialize variables
        self.raster_combos = []
        self.expected_rasters = [
            'Aspect', 'Elevation (DEM)', 'Flow Accumulation', 'Plan Curvature',
            'Profile Curvature', 'Rivers Distance', 'Roads Distance',
            'Slope', 'Stream Power Index', 'Topographic Position Index',
            'Terrain Ruggedness Index', 'Topographic Wetness Index',
            'Lithology', 'Soil'
        ]
        
        self.worker = None
        
        # Setup UI
        self.setup_ui()
        self.connect_signals()
        self.populate_default_paths()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Setup landslide points combo
        self.comboBox_landslide_points.clear()
        self.comboBox_landslide_points.addItem("Select landslide points layer...", None)
        
        # Add vector layers to landslide points combo
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and layer.geometryType() == 0:  # Point geometry
                self.comboBox_landslide_points.addItem(layer.name(), layer.source())
        
        # Setup raster selection area
        self.setup_raster_selection()
        
    def setup_raster_selection(self):
        """Setup raster layer selection combos"""
        
        # Get the scroll area content widget
        content_widget = self.scrollAreaWidgetContents_rasters
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
    
    def connect_signals(self):
        """Connect UI signals"""
        self.pushButton_landslide_browse.clicked.connect(self.browse_landslide_points)
        self.pushButton_output_browse.clicked.connect(self.browse_output_model)
        self.pushButton_start_training.clicked.connect(self.start_training)
        self.pushButton_close.clicked.connect(self.close)
        
    def populate_default_paths(self):
        """Set default paths if available"""
        
        # Set default output path
        plugin_dir = os.path.dirname(__file__)
        default_output = os.path.join(plugin_dir, 'models', 'trained_model.pth')
        self.lineEdit_output_model.setText(default_output)
        
        # Try to auto-select rasters based on layer names
        project_layers = QgsProject.instance().mapLayers()
        
        for combo, expected_name in zip(self.raster_combos, self.expected_rasters):
            for layer_id, layer in project_layers.items():
                if isinstance(layer, QgsRasterLayer):
                    layer_name = layer.name().lower()
                    expected_lower = expected_name.lower()
                    
                    # Simple matching logic
                    if any(keyword in layer_name for keyword in expected_lower.split()):
                        combo.setLayer(layer)
                        break
                        
    def browse_landslide_points(self):
        """Browse for landslide points file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Landslide Points", 
            "", 
            "Vector Files (*.shp *.gpkg *.geojson);;All Files (*)"
        )
        
        if file_path:
            self.comboBox_landslide_points.clear()
            self.comboBox_landslide_points.addItem(os.path.basename(file_path), file_path)
            
    def browse_output_model(self):
        """Browse for output model path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trained Model",
            "",
            "PyTorch Model (*.pth);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.pth'):
                file_path += '.pth'
            self.lineEdit_output_model.setText(file_path)
            
    def validate_inputs(self):
        """Validate user inputs before training"""
        
        # Check landslide points
        landslide_data = self.comboBox_landslide_points.currentData()
        if not landslide_data:
            QMessageBox.warning(self, "Input Error", "Please select landslide points layer.")
            return False
            
        # Check rasters - at least 5 should be selected
        selected_rasters = []
        for combo in self.raster_combos:
            layer = combo.currentLayer()
            if layer:
                selected_rasters.append(layer.source())
                
        if len(selected_rasters) < 5:
            QMessageBox.warning(
                self, "Input Error", 
                f"Please select at least 5 raster layers. Currently selected: {len(selected_rasters)}"
            )
            return False
            
        # Check output path
        output_path = self.lineEdit_output_model.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Input Error", "Please specify output model path.")
            return False
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.warning(self, "Path Error", f"Cannot create output directory: {str(e)}")
                return False
                
        return True, selected_rasters, landslide_data, output_path
        
    def start_training(self):
        """Start the model training process"""
        
        # Validate inputs
        validation_result = self.validate_inputs()
        if validation_result is False:
            return
            
        success, raster_paths, landslide_points_path, output_path = validation_result
        
        # Confirm training start
        reply = QMessageBox.question(
            self, "Start Training", 
            f"Start training with:\n"
            f"• {len(raster_paths)} raster layers\n"
            f"• Landslide points: {os.path.basename(landslide_points_path)}\n"
            f"• Epochs: {self.spinBox_epochs.value()}\n"
            f"• Output: {os.path.basename(output_path)}\n\n"
            f"This process may take several minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Disable training button
        self.pushButton_start_training.setEnabled(False)
        self.pushButton_start_training.setText("Training...")
        
        # Reset progress
        self.progressBar_training.setValue(0)
        self.label_status.setText("Initializing training...")
        
        # Create and start worker thread
        self.worker = ModelTrainingWorker(
            raster_paths=raster_paths,
            landslide_points_path=landslide_points_path,
            output_model_path=output_path,
            epochs=self.spinBox_epochs.value(),
            batch_size=self.spinBox_batch_size.value(),
            test_split=self.doubleSpinBox_test_split.value(),
            generate_non_landslides=self.checkBox_generate_non_landslides.isChecked()
        )
        
        # Connect worker signals
        self.worker.progress.connect(self.progressBar_training.setValue)
        self.worker.status.connect(self.label_status.setText)
        self.worker.finished.connect(self.training_finished)
        
        # Start training
        self.worker.start()
        
    def training_finished(self, success, message):
        """Handle training completion"""
        
        # Re-enable training button
        self.pushButton_start_training.setEnabled(True)
        self.pushButton_start_training.setText("Start Training")
        
        # Show result message
        if success:
            QMessageBox.information(self, "Training Complete", message)
        else:
            QMessageBox.critical(self, "Training Failed", message)
            
        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
