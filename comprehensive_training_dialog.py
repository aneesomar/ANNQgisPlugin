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
    performance_ready = pyqtSignal(dict)  # New signal for performance metrics
    
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
            self.status.emit("Evaluating model performance...")
            
            # Evaluate model on test set using original feature data
            performance_metrics = trainer.evaluate_model_performance(model, feature_data, scaler, selected_features)
            
            self.progress.emit(95)
            self.status.emit("Saving model...")
            
            trainer.save_model(
                model, scaler, selected_features, training_info,
                self.kwargs['output_path']
            )
            
            self.progress.emit(100)
            
            # Emit performance metrics before finishing
            self.performance_ready.emit(performance_metrics)
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
        self.worker.performance_ready.connect(self.show_performance_metrics)
        self.worker.start()
        
    def training_finished(self, success, message):
        """Handle training completion"""
        # Re-enable training button
        self.pushButton_start_training.setEnabled(True)
        self.pushButton_start_training.setText("Start Training")
        
        # Store the completion message for later
        self.completion_message = message
        self.training_success = success
        
        # If training failed, show error immediately
        if not success:
            QMessageBox.critical(self, "Training Failed", message)
            
        # Performance metrics dialog will be shown by show_performance_metrics
        # and then the completion message will be shown
            
        # Clean up
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
            
    def show_performance_metrics(self, metrics):
        """Display model performance metrics in a dialog"""
        dialog = PerformanceDialog(metrics, self)
        dialog.exec_()
        
        # After performance dialog is closed, show completion message
        if hasattr(self, 'training_success') and self.training_success:
            QMessageBox.information(self, "Training Complete", self.completion_message)


class PerformanceDialog(QtWidgets.QDialog):
    """Dialog to display model performance metrics"""
    
    def __init__(self, metrics, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the performance display UI"""
        self.setWindowTitle("Model Performance Metrics")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎯 Model Training Performance Results")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E8B57; margin: 10px;")
        layout.addWidget(title)
        
        # Scroll area for content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QtWidgets.QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Overall metrics section
        metrics_group = QtWidgets.QGroupBox("📊 Overall Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Create metrics display
        accuracy = self.metrics.get('accuracy', 0) * 100
        precision = self.metrics.get('precision', 0) * 100
        recall = self.metrics.get('recall', 0) * 100
        f1_score = self.metrics.get('f1_score', 0) * 100
        
        metrics_text = f"""
<div style="font-family: monospace; line-height: 1.6;">
<b>🎯 Accuracy:</b> <span style="color: #2E8B57; font-size: 14px;"><b>{accuracy:.2f}%</b></span><br>
<b>🎯 Precision:</b> <span style="color: #4169E1; font-size: 14px;"><b>{precision:.2f}%</b></span><br>
<b>🎯 Recall:</b> <span style="color: #FF6347; font-size: 14px;"><b>{recall:.2f}%</b></span><br>
<b>🎯 F1-Score:</b> <span style="color: #9932CC; font-size: 14px;"><b>{f1_score:.2f}%</b></span><br>
</div>
        """
        
        metrics_label = QLabel(metrics_text)
        metrics_label.setWordWrap(True)
        metrics_layout.addWidget(metrics_label)
        content_layout.addWidget(metrics_group)
        
        # Predictions distribution
        pred_group = QtWidgets.QGroupBox("📈 Predictions Distribution")
        pred_layout = QVBoxLayout(pred_group)
        
        pred_dist = self.metrics.get('predictions_distribution', {})
        pred_text = f"""
<div style="font-family: monospace; line-height: 1.6;">
<b>Test Set Size:</b> {self.metrics.get('test_size', 0)} samples<br><br>
<b>Predicted Results:</b><br>
• Landslides: {pred_dist.get('predicted_landslides', 0)}<br>
• Non-landslides: {pred_dist.get('predicted_non_landslides', 0)}<br><br>
<b>Actual Ground Truth:</b><br>
• Landslides: {pred_dist.get('actual_landslides', 0)}<br>
• Non-landslides: {pred_dist.get('actual_non_landslides', 0)}<br>
</div>
        """
        
        pred_label = QLabel(pred_text)
        pred_label.setWordWrap(True)
        pred_layout.addWidget(pred_label)
        content_layout.addWidget(pred_group)
        
        # Confusion matrix
        cm_group = QtWidgets.QGroupBox("🔍 Confusion Matrix")
        cm_layout = QVBoxLayout(cm_group)
        
        cm = self.metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        cm_text = f"""
<div style="font-family: monospace; line-height: 1.8;">
<table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #f0f0f0;">
    <th></th><th>Predicted Non-Landslide</th><th>Predicted Landslide</th>
</tr>
<tr>
    <td style="background-color: #f0f0f0;"><b>Actual Non-Landslide</b></td>
    <td style="text-align: center; color: green;"><b>{cm[0][0]}</b></td>
    <td style="text-align: center; color: red;"><b>{cm[0][1]}</b></td>
</tr>
<tr>
    <td style="background-color: #f0f0f0;"><b>Actual Landslide</b></td>
    <td style="text-align: center; color: red;"><b>{cm[1][0]}</b></td>
    <td style="text-align: center; color: green;"><b>{cm[1][1]}</b></td>
</tr>
</table>
</div>
        """
        
        cm_label = QLabel(cm_text)
        cm_label.setWordWrap(True)
        cm_layout.addWidget(cm_label)
        content_layout.addWidget(cm_group)
        
        # Interpretation section
        interp_group = QtWidgets.QGroupBox("💡 Performance Interpretation")
        interp_layout = QVBoxLayout(interp_group)
        
        # Generate interpretation
        interpretation = self.get_performance_interpretation()
        interp_label = QLabel(interpretation)
        interp_label.setWordWrap(True)
        interp_layout.addWidget(interp_label)
        content_layout.addWidget(interp_group)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
    def get_performance_interpretation(self):
        """Generate interpretation of the performance metrics"""
        accuracy = self.metrics.get('accuracy', 0) * 100
        precision = self.metrics.get('precision', 0) * 100
        recall = self.metrics.get('recall', 0) * 100
        f1_score = self.metrics.get('f1_score', 0) * 100
        
        interpretation = "<div style='line-height: 1.6;'>"
        
        # Overall assessment
        if accuracy >= 90:
            interpretation += "🟢 <b>Excellent Performance:</b> Your model shows exceptional accuracy.<br>"
        elif accuracy >= 80:
            interpretation += "🟡 <b>Good Performance:</b> Your model performs well with room for improvement.<br>"
        elif accuracy >= 70:
            interpretation += "🟠 <b>Moderate Performance:</b> Consider collecting more data or feature engineering.<br>"
        else:
            interpretation += "🔴 <b>Poor Performance:</b> Model needs significant improvement.<br>"
            
        interpretation += "<br>"
        
        # Precision interpretation
        if precision >= 85:
            interpretation += "✅ <b>High Precision:</b> Low false positive rate - reliable landslide predictions.<br>"
        else:
            interpretation += "⚠️ <b>Lower Precision:</b> Some non-landslide areas may be incorrectly classified.<br>"
            
        # Recall interpretation  
        if recall >= 85:
            interpretation += "✅ <b>High Recall:</b> Good at detecting actual landslides.<br>"
        else:
            interpretation += "⚠️ <b>Lower Recall:</b> Some actual landslides may be missed.<br>"
            
        interpretation += "<br><b>Recommendations:</b><br>"
        if accuracy < 80:
            interpretation += "• Consider adding more training data<br>"
            interpretation += "• Review and add more relevant raster layers<br>"
            interpretation += "• Check data quality and landslide point accuracy<br>"
        else:
            interpretation += "• Model is ready for practical use<br>"
            interpretation += "• Consider validation with field data<br>"
            
        interpretation += "</div>"
        return interpretation
