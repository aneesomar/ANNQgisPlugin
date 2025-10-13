#!/usr/bin/env python3
"""
Validation Module for Landslide Predictions
Validates prediction raster against known landslide locations
Calculates accuracy, precision, recall, F1, ROC-AUC, etc.
"""

import numpy as np
from osgeo import gdal, ogr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import os
import sys


def validate_prediction_against_points(prediction_raster_path, 
                                       validation_points_path,
                                       threshold=0.65,
                                       verbose=True):
    """
    Validate prediction raster against known landslide points
    
    Args:
        prediction_raster_path: Path to susceptibility prediction raster
        validation_points_path: Path to validation points (shapefile or CSV with x,y,landslide columns)
        threshold: Threshold for binary classification (default 0.65)
        verbose: Print detailed output
    
    Returns:
        dict with metrics
    """
    
    if verbose:
        print("=" * 70)
        print("PREDICTION VALIDATION")
        print("=" * 70)
        print(f"\n📁 Prediction raster: {prediction_raster_path}")
        print(f"📍 Validation points: {validation_points_path}")
        print(f"🎯 Threshold: {threshold}")
    
    # Open prediction raster
    pred_ds = gdal.Open(prediction_raster_path)
    if pred_ds is None:
        raise ValueError(f"Could not open prediction raster: {prediction_raster_path}")
    
    pred_band = pred_ds.GetRasterBand(1)
    pred_array = pred_band.ReadAsArray()
    geotransform = pred_ds.GetGeoTransform()
    
    # Load validation points
    if validation_points_path.endswith('.csv'):
        validation_data = load_validation_csv(validation_points_path, verbose)
    elif validation_points_path.endswith('.shp'):
        validation_data = load_validation_shapefile(validation_points_path, verbose)
    else:
        raise ValueError("Validation file must be .csv or .shp")
    
    if verbose:
        print(f"\n📊 Loaded {len(validation_data)} validation points")
        landslide_count = sum(1 for _, _, label in validation_data if label == 1)
        print(f"   Landslides: {landslide_count} ({landslide_count/len(validation_data)*100:.1f}%)")
        print(f"   Non-landslides: {len(validation_data) - landslide_count} ({(len(validation_data) - landslide_count)/len(validation_data)*100:.1f}%)")
    
    # Extract predictions at validation points
    y_true = []
    y_pred_proba = []
    valid_points = 0
    
    for x, y, label in validation_data:
        # Convert geographic coordinates to pixel coordinates
        px = int((x - geotransform[0]) / geotransform[1])
        py = int((y - geotransform[3]) / geotransform[5])
        
        # Check if within raster bounds
        if 0 <= px < pred_ds.RasterXSize and 0 <= py < pred_ds.RasterYSize:
            pred_value = pred_array[py, px]
            
            # Skip if NoData
            if not np.isnan(pred_value) and pred_value != pred_band.GetNoDataValue():
                y_true.append(label)
                y_pred_proba.append(pred_value)
                valid_points += 1
    
    if verbose:
        print(f"   Valid predictions: {valid_points}/{len(validation_data)} ({valid_points/len(validation_data)*100:.1f}%)")
    
    if valid_points == 0:
        raise ValueError("No valid predictions at validation points!")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION METRICS")
        print("=" * 70)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC-AUC
    if len(np.unique(y_true)) > 1:  # Need both classes for ROC-AUC
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['roc_auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    metrics['confusion_matrix'] = cm
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
    
    # Prediction distribution
    metrics['pred_mean'] = y_pred_proba.mean()
    metrics['pred_std'] = y_pred_proba.std()
    metrics['pred_min'] = y_pred_proba.min()
    metrics['pred_max'] = y_pred_proba.max()
    
    # Print results
    if verbose:
        print(f"\n📈 Classification Metrics (threshold = {threshold}):")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0        1")
        print(f"   Actual 0   {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"   Actual 1   {cm[1,0]:6d}   {cm[1,1]:6d}")
        
        if cm.shape == (2, 2):
            print(f"\n   True Negatives:  {tn:6d}")
            print(f"   False Positives: {fp:6d}")
            print(f"   False Negatives: {fn:6d}")
            print(f"   True Positives:  {tp:6d}")
        
        print(f"\n📉 Prediction Distribution:")
        print(f"   Mean: {metrics['pred_mean']:.4f}")
        print(f"   Std:  {metrics['pred_std']:.4f}")
        print(f"   Min:  {metrics['pred_min']:.4f}")
        print(f"   Max:  {metrics['pred_max']:.4f}")
        
        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)
        
        if metrics['accuracy'] >= 0.75:
            print("✅ Accuracy: Excellent (≥ 75%)")
        elif metrics['accuracy'] >= 0.65:
            print("🟡 Accuracy: Good (65-75%)")
        else:
            print("🔴 Accuracy: Poor (< 65%)")
        
        if metrics['recall'] >= 0.75:
            print("✅ Recall: Good landslide detection (≥ 75%)")
        elif metrics['recall'] >= 0.60:
            print("🟡 Recall: Moderate landslide detection (60-75%)")
        else:
            print("🔴 Recall: Missing many landslides (< 60%)")
        
        if metrics['precision'] >= 0.70:
            print("✅ Precision: Low false alarms (≥ 70%)")
        elif metrics['precision'] >= 0.50:
            print("🟡 Precision: Moderate false alarms (50-70%)")
        else:
            print("🔴 Precision: High false alarms (< 50%)")
        
        if metrics['roc_auc'] and metrics['roc_auc'] >= 0.80:
            print("✅ ROC-AUC: Excellent discrimination (≥ 0.80)")
        elif metrics['roc_auc'] and metrics['roc_auc'] >= 0.70:
            print("🟡 ROC-AUC: Good discrimination (0.70-0.80)")
        elif metrics['roc_auc']:
            print("🔴 ROC-AUC: Poor discrimination (< 0.70)")
        
        print("\n" + "=" * 70)
    
    return metrics


def load_validation_csv(csv_path, verbose=True):
    """
    Load validation data from CSV
    Expected columns: x, y, landslide (or similar)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"\n   CSV columns: {list(df.columns)}")
    
    # Try to find coordinate columns
    x_col = None
    y_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'x' in col_lower and 'lon' not in col_lower:
            x_col = col
        elif 'y' in col_lower and 'lat' not in col_lower:
            y_col = col
        elif 'landslide' in col_lower or 'label' in col_lower or 'class' in col_lower:
            label_col = col
    
    if x_col is None or y_col is None or label_col is None:
        raise ValueError(f"Could not find x, y, landslide columns in CSV. Available: {list(df.columns)}")
    
    if verbose:
        print(f"   Using: X={x_col}, Y={y_col}, Label={label_col}")
    
    validation_data = []
    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        label = int(row[label_col])
        validation_data.append((x, y, label))
    
    return validation_data


def load_validation_shapefile(shp_path, verbose=True):
    """
    Load validation data from shapefile
    Expected attribute: landslide (or similar)
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shp_path, 0)
    
    if datasource is None:
        raise ValueError(f"Could not open shapefile: {shp_path}")
    
    layer = datasource.GetLayer()
    
    # Find landslide attribute
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    
    if verbose:
        print(f"\n   Shapefile fields: {field_names}")
    
    label_field = None
    for field in field_names:
        if 'landslide' in field.lower() or 'label' in field.lower() or 'class' in field.lower():
            label_field = field
            break
    
    if label_field is None:
        raise ValueError(f"Could not find landslide field in shapefile. Available: {field_names}")
    
    if verbose:
        print(f"   Using label field: {label_field}")
    
    validation_data = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()
        label = int(feature.GetField(label_field))
        validation_data.append((x, y, label))
    
    datasource = None
    return validation_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate landslide susceptibility predictions against known points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validate_prediction.py prediction.tif validation.csv
  python3 validate_prediction.py prediction.tif validation.shp --threshold 0.7

CSV Format:
  Must have columns: x, y, landslide (0 or 1)

Shapefile Format:
  Must have attribute: landslide (0 or 1)
        """
    )
    
    parser.add_argument('prediction', help='Path to prediction raster')
    parser.add_argument('validation', help='Path to validation points (.csv or .shp)')
    parser.add_argument('--threshold', type=float, default=0.65, 
                       help='Classification threshold (default: 0.65)')
    
    args = parser.parse_args()
    
    try:
        metrics = validate_prediction_against_points(
            args.prediction,
            args.validation,
            threshold=args.threshold,
            verbose=True
        )
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
