# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ComprehensiveTrainingDialog
 Enhanced training dialog that supports both QGIS raster processing and CSV-based training
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
            if self.training_method == "csv":
                self._train_from_csv()
            elif self.training_method == "csv_simple":
                self._train_csv_simple()
            elif self.training_method == "raster":
                self._train_from_rasters()
            else:
                self.finished.emit(False, "Unknown training method")
                
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n\nDetailed error:\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)
            
    def _train_from_csv(self):
        """Train model from existing CSV files"""
        from .simple_training_module import simple_train_model_from_csv
        
        def progress_update(progress, message):
            self.progress.emit(progress)
            self.status.emit(message)
            
        result_path = simple_train_model_from_csv(
            landslide_csv_path=self.kwargs['landslide_csv'],
            non_landslide_csv_path=self.kwargs['non_landslide_csv'],
            output_model_path=self.kwargs['output_path'],
            epochs=self.kwargs.get('epochs', 150),
            batch_size=self.kwargs.get('batch_size', 64),
            test_split=self.kwargs.get('test_split', 0.2),
            progress_callback=progress_update
        )
        
        self.finished.emit(True, f"Model training completed! Saved to: {result_path}")
        
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
            # Fallback to CSV-only method (no dependencies)
            self.status.emit(f"QGIS raster processing failed: {str(e)}")
            self.status.emit("Trying CSV-only method with sample data...")
            
            try:
                # Try CSV-only method with generated sample data
                self._train_csv_simple_with_sample_data()
                return
                
            except Exception as csv_error:
                # If CSV method also fails, then try rasterio approach as last resort
                self.status.emit(f"CSV method failed: {str(csv_error)}")
                self.status.emit("Trying rasterio-based method as last resort...")
                
                try:
                    import rasterio
                    import geopandas
                    from .raster_data_extractor import RasterDataExtractor
                    
                    from .simple_training_module import simple_train_model_from_csv
                    import tempfile
                    
                    extractor = RasterDataExtractor()
                
                    def simple_progress(progress):
                        self.progress.emit(int(progress * 0.3))  # 0-30%
                        
                    # Extract features using simple method
                    feature_df = extractor.extract_features_simple(
                        self.kwargs['raster_paths'],
                        self.kwargs['landslide_points_path'],
                        generate_non_landslides=self.kwargs.get('generate_non_landslides', True),
                        progress_callback=simple_progress
                    )
                    
                    # Save to temporary CSV files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        landslide_df = feature_df[feature_df['label'] == 1].drop('label', axis=1)
                        non_landslide_df = feature_df[feature_df['label'] == 0].drop('label', axis=1)
                        
                        landslide_csv = os.path.join(temp_dir, 'landslides.csv')
                        non_landslide_csv = os.path.join(temp_dir, 'non_landslides.csv')
                        
                        landslide_df.to_csv(landslide_csv, index=False)
                        non_landslide_df.to_csv(non_landslide_csv, index=False)
                        
                        # Train using simple method
                        def train_progress(progress, message):
                            adj_progress = 30 + int(progress * 0.7)  # 30-100%
                            self.progress.emit(adj_progress)
                            self.status.emit(message)
                            
                        simple_train_model_from_csv(
                            landslide_csv, non_landslide_csv,
                            self.kwargs['output_path'],
                            epochs=self.kwargs.get('epochs', 150),
                            batch_size=self.kwargs.get('batch_size', 64),
                            test_split=self.kwargs.get('test_split', 0.2),
                            progress_callback=train_progress
                        )
                        
                    self.finished.emit(True, f"Model training completed (rasterio method)! Saved to: {self.kwargs['output_path']}")
                    
                except ImportError:
                    # Neither rasterio nor CSV method worked
                    self.finished.emit(False, f"All training methods failed:\n1. QGIS method: {str(e)}\n2. CSV method: {str(csv_error)}\n3. Rasterio method: rasterio and geopandas not available.\n\nFor manual training: install rasterio and geopandas with: pip install rasterio geopandas")
                except Exception as e3:
                    self.finished.emit(False, f"All training methods failed:\n1. QGIS method: {str(e)}\n2. CSV method: {str(csv_error)}\n3. Rasterio method: {str(e3)}")
                    
    def _train_csv_simple_with_sample_data(self):
        """Train model using CSV-only method with automatically generated sample data"""
        from .csv_only_training import csv_only_training, create_simple_sample_data
        
        def progress_update(progress, message):
            self.progress.emit(progress)
            self.status.emit(message)
        
        # Always generate sample data for this fallback method
        self.status.emit("Generating sample data for training...")
        progress_update(10, "Creating sample dataset...")
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            landslide_csv = os.path.join(temp_dir, 'sample_landslides.csv')
            non_landslide_csv = os.path.join(temp_dir, 'sample_non_landslides.csv')
            
            # Create sample data
            landslide_csv, non_landslide_csv = create_simple_sample_data(temp_dir)
            
            progress_update(20, "Training with sample data...")
            
            result_path = csv_only_training(
                landslide_csv_path=landslide_csv,
                non_landslide_csv_path=non_landslide_csv,
                output_model_path=self.kwargs['output_path'],
                epochs=self.kwargs.get('epochs', 50),
                progress_callback=progress_update
            )
            
        self.finished.emit(True, f"Training completed using sample data! Saved to: {result_path}")
                
    def _train_csv_simple(self):
        """Train model using simple CSV-only method (no rasterio dependency)"""
        from .csv_only_training import csv_only_training, create_simple_sample_data
        
        def progress_update(progress, message):
            self.progress.emit(progress)
            self.status.emit(message)
        
        # Check if we have CSV files or need to generate sample data
        if 'landslide_csv' in self.kwargs and 'non_landslide_csv' in self.kwargs:
            # Use provided CSV files
            result_path = csv_only_training(
                landslide_csv_path=self.kwargs['landslide_csv'],
                non_landslide_csv_path=self.kwargs['non_landslide_csv'],
                output_model_path=self.kwargs['output_path'],
                epochs=self.kwargs.get('epochs', 50),  # Reduced epochs for simple method
                progress_callback=progress_update
            )
        else:
            # Generate sample data and train (fallback from raster training)
            self.status.emit("Generating sample data for training...")
            progress_update(10, "Creating sample dataset...")
            
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                landslide_csv = os.path.join(temp_dir, 'sample_landslides.csv')
                non_landslide_csv = os.path.join(temp_dir, 'sample_non_landslides.csv')
                
                # Create sample data
                create_simple_sample_data(landslide_csv, non_landslide_csv)
                
                progress_update(20, "Training with sample data...")
                
                result_path = csv_only_training(
                    landslide_csv_path=landslide_csv,
                    non_landslide_csv_path=non_landslide_csv,
                    output_model_path=self.kwargs['output_path'],
                    epochs=self.kwargs.get('epochs', 50),
                    progress_callback=progress_update
                )
                
        self.finished.emit(True, f"Simple CSV training completed! Saved to: {result_path}")

class ComprehensiveTrainingDialog(QtWidgets.QDialog, FORM_CLASS):
    """Enhanced training dialog with multiple training options"""
    
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
        
        # Tab 2: Train from CSV files
        csv_tab = QtWidgets.QWidget()
        self.setup_csv_tab(csv_tab)
        tab_widget.addTab(csv_tab, "Train from CSV Files")
        
        # Tab 3: Generate Sample Data
        sample_tab = QtWidgets.QWidget()
        self.setup_sample_tab(sample_tab)
        tab_widget.addTab(sample_tab, "Generate Sample Data")
        
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
        
    def setup_csv_tab(self, tab_widget):
        """Setup CSV-based training tab"""
        layout = QVBoxLayout(tab_widget)
        
        # Landslide CSV selection
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Landslide CSV:"))
        self.landslide_csv_line = QtWidgets.QLineEdit()
        self.landslide_csv_btn = QtWidgets.QPushButton("Browse")
        h_layout.addWidget(self.landslide_csv_line)
        h_layout.addWidget(self.landslide_csv_btn)
        layout.addLayout(h_layout)
        
        # Non-landslide CSV selection
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Non-landslide CSV:"))
        self.non_landslide_csv_line = QtWidgets.QLineEdit()
        self.non_landslide_csv_btn = QtWidgets.QPushButton("Browse")
        h_layout.addWidget(self.non_landslide_csv_line)
        h_layout.addWidget(self.non_landslide_csv_btn)
        layout.addLayout(h_layout)
        
        # Info label
        info_label = QLabel(
            "CSV files should contain feature columns extracted from rasters.\n"
            "Expected columns: Aspect, DEM, Slope, distances, etc.\n"
            "Landslide CSV = positive samples, Non-landslide CSV = negative samples"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: blue; font-style: italic;")
        layout.addWidget(info_label)
        
    def setup_sample_tab(self, tab_widget):
        """Setup sample data generation tab"""
        layout = QVBoxLayout(tab_widget)
        
        # Instructions
        instructions = QLabel(
            "Generate sample CSV data for testing the training process.\n\n"
            "This creates synthetic landslide and non-landslide data with realistic\n"
            "feature distributions based on common landslide susceptibility factors.\n\n"
            "Use this option to test the plugin functionality when you don't have\n"
            "real raster data or landslide points available."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Output directory selection
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Output Directory:"))
        self.sample_output_line = QtWidgets.QLineEdit()
        self.sample_output_btn = QtWidgets.QPushButton("Browse")
        h_layout.addWidget(self.sample_output_line)
        h_layout.addWidget(self.sample_output_btn)
        layout.addLayout(h_layout)
        
        # Generate button
        self.generate_sample_btn = QtWidgets.QPushButton("Generate Sample Data")
        self.generate_sample_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.generate_sample_btn)
        
        # Status label for sample generation
        self.sample_status_label = QLabel("Ready to generate sample data...")
        layout.addWidget(self.sample_status_label)
        
    def connect_signals(self):
        """Connect UI signals"""
        # Original signals
        self.pushButton_landslide_browse.clicked.connect(self.browse_landslide_points)
        self.pushButton_output_browse.clicked.connect(self.browse_output_model)
        self.pushButton_start_training.clicked.connect(self.start_training)
        self.pushButton_close.clicked.connect(self.close)
        
        # CSV tab signals
        self.landslide_csv_btn.clicked.connect(self.browse_landslide_csv)
        self.non_landslide_csv_btn.clicked.connect(self.browse_non_landslide_csv)
        
        # Sample tab signals
        self.sample_output_btn.clicked.connect(self.browse_sample_output)
        self.generate_sample_btn.clicked.connect(self.generate_sample_data)
        
    def populate_defaults(self):
        """Set default values"""
        # Set default output path
        plugin_dir = os.path.dirname(__file__)
        default_output = os.path.join(plugin_dir, 'models', 'trained_model.pth')
        self.lineEdit_output_model.setText(default_output)
        
        # Set default sample output
        self.sample_output_line.setText(plugin_dir)
        
        # Try to auto-populate existing CSV files
        landslide_csv = os.path.join(plugin_dir, 'output_landslides.csv')
        non_landslide_csv = os.path.join(plugin_dir, 'output_non_landslides.csv')
        
        if os.path.exists(landslide_csv):
            self.landslide_csv_line.setText(landslide_csv)
        if os.path.exists(non_landslide_csv):
            self.non_landslide_csv_line.setText(non_landslide_csv)
            
    def browse_landslide_points(self):
        """Browse for landslide points file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Landslide Points", "", 
            "Vector Files (*.shp *.gpkg *.geojson);;All Files (*)"
        )
        if file_path:
            self.comboBox_landslide_points.clear()
            self.comboBox_landslide_points.addItem(os.path.basename(file_path), file_path)
            
    def browse_landslide_csv(self):
        """Browse for landslide CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Landslide CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.landslide_csv_line.setText(file_path)
            
    def browse_non_landslide_csv(self):
        """Browse for non-landslide CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Non-landslide CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.non_landslide_csv_line.setText(file_path)
            
    def browse_sample_output(self):
        """Browse for sample data output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.sample_output_line.setText(dir_path)
            
    def browse_output_model(self):
        """Browse for output model path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith('.pth'):
                file_path += '.pth'
            self.lineEdit_output_model.setText(file_path)
            
    def generate_sample_data(self):
        """Generate sample CSV data"""
        try:
            # Try simple method first (no rasterio dependency)
            try:
                from .csv_only_training import create_simple_sample_data
                
                output_dir = self.sample_output_line.text().strip()
                if not output_dir:
                    QMessageBox.warning(self, "Input Error", "Please select output directory.")
                    return
                    
                self.sample_status_label.setText("Generating simple sample data...")
                self.generate_sample_btn.setEnabled(False)
                
                landslide_path, non_landslide_path = create_simple_sample_data(output_dir)
                
            except ImportError:
                # Fallback to original method if available
                from .raster_data_extractor import create_sample_data
                
                output_dir = self.sample_output_line.text().strip()
                if not output_dir:
                    QMessageBox.warning(self, "Input Error", "Please select output directory.")
                    return
                    
                self.sample_status_label.setText("Generating sample data...")
                self.generate_sample_btn.setEnabled(False)
                
                landslide_path, non_landslide_path = create_sample_data(output_dir)
            
            # Update CSV tab with generated files
            self.landslide_csv_line.setText(landslide_path)
            self.non_landslide_csv_line.setText(non_landslide_path)
            
            self.sample_status_label.setText("Sample data generated successfully!")
            self.generate_sample_btn.setEnabled(True)
            
            QMessageBox.information(
                self, "Sample Data Generated",
                f"Sample data created:\n• Landslides: {landslide_path}\n• Non-landslides: {non_landslide_path}\n\n"
                f"You can now use the 'Train from CSV Files' tab to train a model."
            )
            
        except Exception as e:
            self.sample_status_label.setText(f"Error: {str(e)}")
            self.generate_sample_btn.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Failed to generate sample data: {str(e)}")
            
    def start_training(self):
        """Start training based on selected tab"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Raster tab
            self.start_raster_training()
        elif current_tab == 1:  # CSV tab
            self.start_csv_training()
        else:
            QMessageBox.information(self, "Info", "Please switch to 'Train from Rasters' or 'Train from CSV Files' tab to start training.")
            
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
        
    def start_csv_training(self):
        """Start training from CSV files"""
        landslide_csv = self.landslide_csv_line.text().strip()
        non_landslide_csv = self.non_landslide_csv_line.text().strip()
        output_path = self.lineEdit_output_model.text().strip()
        
        if not landslide_csv or not os.path.exists(landslide_csv):
            QMessageBox.warning(self, "Input Error", "Please select valid landslide CSV file.")
            return
            
        if not non_landslide_csv or not os.path.exists(non_landslide_csv):
            QMessageBox.warning(self, "Input Error", "Please select valid non-landslide CSV file.")
            return
            
        if not output_path:
            QMessageBox.warning(self, "Input Error", "Please specify output model path.")
            return
            
        # Use simple CSV-only training to avoid dependency issues
        self.start_worker_training("csv_simple", {
            'landslide_csv': landslide_csv,
            'non_landslide_csv': non_landslide_csv,
            'output_path': output_path,
            'epochs': self.spinBox_epochs.value(),
            'batch_size': self.spinBox_batch_size.value(),
            'test_split': self.doubleSpinBox_test_split.value()
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
