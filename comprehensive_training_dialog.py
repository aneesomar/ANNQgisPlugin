# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ComprehensiveTrainingDialog
 Enhanced training dialog for QGIS raster processing-based landslide model training
 ***************************************************************************/
"""

import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import QThread, pyqtSignal, QTimer
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget
from qgis.gui import QgsMapLayerComboBox, QgsFileWidget
from qgis.core import QgsMapLayerProxyModel, QgsVectorLayer, QgsRasterLayer, QgsProject

# Load the UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'model_training_dialog_base.ui'))

class ComprehensiveTrainingWorker(QThread):
    """Worker thread for model training with multiple methods"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, training_method, **kwargs):
        super().__init__()
        self.training_method = training_method
        self.kwargs = kwargs
        
    def run(self):
        """Run the training process based on selected method"""
        try:
            if self.training_method == "raster":
                self._train_from_rasters()
            else:
                self.finished.emit(False, "Unknown training method")
                
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n\nDetailed error:\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)
            

    def _train_from_rasters(self):
        """Train model from raster data extraction"""
        # Try QGIS-based approach first, fall back to simple approach
        try:
            from .ann_training_module import ANNTrainingModule
            trainer = ANNTrainingModule()
            
            # Extract features
            self.status.emit("Extracting features from rasters using QGIS...")
            
            def feature_progress(progress):
                self.progress.emit(int(progress * 0.4))  # 0-40%
                
            feature_data = trainer.extract_features_from_rasters(
                self.kwargs['raster_paths'],
                self.kwargs['landslide_points_path'],
                generate_non_landslides=self.kwargs.get('generate_non_landslides', True),
                progress_callback=feature_progress
            )
            
            self.progress.emit(40)
            self.status.emit("Preparing training data...")
            
            X, y, selected_features, scaler = trainer.prepare_training_data(
                feature_data, 
                test_split=self.kwargs.get('test_split', 0.2)
            )
            
            self.progress.emit(50)
            
            def training_progress(epoch, total_epochs):
                progress = 50 + int((epoch / total_epochs) * 40)  # 50-90%
                self.progress.emit(progress)
                self.status.emit(f"Training epoch {epoch}/{total_epochs}")
                
            model, training_info = trainer.train_model(
                X, y,
                epochs=self.kwargs.get('epochs', 150),
                batch_size=self.kwargs.get('batch_size', 64),
                progress_callback=training_progress
            )
            
            self.progress.emit(90)
            self.status.emit("Saving model...")
            
            trainer.save_model(
                model, scaler, selected_features, training_info,
                self.kwargs['output_path']
            )
            
            self.progress.emit(100)
            self.finished.emit(True, f"Advanced model training completed! Saved to: {self.kwargs['output_path']}")
            
        except Exception as e:
            # If raster processing fails, report the error
            self.finished.emit(False, f"Raster training failed: {str(e)}")


class ComprehensiveTrainingDialog(QtWidgets.QDialog, FORM_CLASS):
    """Enhanced training dialog for raster-based landslide model training"""
    
    def __init__(self, parent=None):
        super(ComprehensiveTrainingDialog, self).__init__(parent)
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
        
        # Setup UI with tabs for different training methods
        self.setup_tabbed_ui()
        self.connect_signals()
        self.populate_defaults()
        
    def setup_tabbed_ui(self):
        """Setup tabbed interface for different training methods"""
        
        # Create tab widget to replace the main content
        tab_widget = QTabWidget()
        
        # Tab 1: Train from Rasters (original interface)
        raster_tab = QtWidgets.QWidget()
        self.setup_raster_tab(raster_tab)
        tab_widget.addTab(raster_tab, "Train from Rasters")
        
        # Replace the scroll area with tabs
        # First, get the current position of scroll area
        scroll_geom = self.scrollArea_rasters.geometry()
        
        # Remove the scroll area and add tab widget
        self.scrollArea_rasters.setParent(None)
        tab_widget.setParent(self)
        tab_widget.setGeometry(scroll_geom.x(), scroll_geom.y(), 
                               scroll_geom.width(), scroll_geom.height() - 50)
        
        self.tab_widget = tab_widget
        
    def setup_raster_tab(self, tab_widget):
        """Setup raster-based training tab"""
        layout = QVBoxLayout(tab_widget)
        
        # Landslide points selection
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Landslide Points:"))
        
        self.raster_landslide_combo = QgsMapLayerComboBox()
        self.raster_landslide_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        h_layout.addWidget(self.raster_landslide_combo)
        
        layout.addLayout(h_layout)
        
        # Raster selection area
        layout.addWidget(QLabel("Select Raster Layers:"))
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Create combo boxes for each expected raster
        self.raster_combos = []
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
            scroll_layout.addLayout(h_layout)
            
            self.raster_combos.append(combo)
            
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
    def connect_signals(self):
        """Connect UI signals"""
        # Original signals
        self.pushButton_landslide_browse.clicked.connect(self.browse_landslide_points)
        self.pushButton_output_browse.clicked.connect(self.browse_output_model)
        self.pushButton_start_training.clicked.connect(self.start_training)
        self.pushButton_close.clicked.connect(self.close)
        
    def populate_defaults(self):
        """Set default values"""
        # Set default output path
        plugin_dir = os.path.dirname(__file__)
        default_output = os.path.join(plugin_dir, 'models', 'trained_model.pth')
        self.lineEdit_output_model.setText(default_output)
            
    def browse_landslide_points(self):
        """Browse for landslide points file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Landslide Points", "", 
            "Vector Files (*.shp *.gpkg *.geojson);;All Files (*)"
        )
        if file_path:
            self.comboBox_landslide_points.clear()
            self.comboBox_landslide_points.addItem(os.path.basename(file_path), file_path)
            
    def browse_output_model(self):
        """Browse for output model path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith('.pth'):
                file_path += '.pth'
            self.lineEdit_output_model.setText(file_path)
            
    def start_training(self):
        """Start training based on selected tab"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Raster tab (the only remaining tab)
            self.start_raster_training()
        else:
            QMessageBox.information(self, "Info", "Please use the 'Train from Rasters' tab to start training.")
            
    def start_raster_training(self):
        """Start training from rasters"""
        # Validate raster inputs
        landslide_layer = self.raster_landslide_combo.currentLayer()
        if not landslide_layer:
            QMessageBox.warning(self, "Input Error", "Please select landslide points layer.")
            return
            
        selected_rasters = []
        for combo in self.raster_combos:
            layer = combo.currentLayer()
            if layer:
                selected_rasters.append(layer.source())
                
        if len(selected_rasters) < 3:
            QMessageBox.warning(self, "Input Error", f"Please select at least 3 raster layers. Currently selected: {len(selected_rasters)}")
            return
            
        output_path = self.lineEdit_output_model.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Input Error", "Please specify output model path.")
            return
            
        # Start training
        self.start_worker_training("raster", {
            'raster_paths': selected_rasters,
            'landslide_points_path': landslide_layer.source(),
            'output_path': output_path,
            'epochs': self.spinBox_epochs.value(),
            'batch_size': self.spinBox_batch_size.value(),
            'test_split': self.doubleSpinBox_test_split.value(),
            'generate_non_landslides': self.checkBox_generate_non_landslides.isChecked()
        })
        
    def start_worker_training(self, method, kwargs):
        """Start worker thread for training"""
        # Disable training button
        self.pushButton_start_training.setEnabled(False)
        self.pushButton_start_training.setText("Training...")
        
        # Reset progress
        self.progressBar_training.setValue(0)
        self.label_status.setText("Initializing training...")
        
        # Create and start worker
        self.worker = ComprehensiveTrainingWorker(method, **kwargs)
        self.worker.progress.connect(self.progressBar_training.setValue)
        self.worker.status.connect(self.label_status.setText)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()
        
    def training_finished(self, success, message):
        """Handle training completion"""
        # Re-enable training button
        self.pushButton_start_training.setEnabled(True)
        self.pushButton_start_training.setText("Start Training")
        
        # Show result
        if success:
            QMessageBox.information(self, "Training Complete", message)
        else:
            QMessageBox.critical(self, "Training Failed", message)
            
        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
