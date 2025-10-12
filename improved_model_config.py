# Improved Model Configuration
# Based on comprehensive analysis findings

CONFIG = {'model': {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.5, 'weight_decay': 0.01}, 'training': {'learning_rate': 0.0001, 'batch_size': 64, 'max_epochs': 100, 'patience': 10, 'focal_loss': {'alpha': 0.25, 'gamma': 2.0}}, 'data_balancing': {'use_smote': True, 'smote_k_neighbors': 5, 'class_weight_balancing': True}, 'threshold_optimization': {'search_range': [0.3, 0.4, 0.5, 0.6, 0.7], 'metric': 'f1_score'}, 'spatial_validation': {'use_spatial_cv': True, 'n_folds': 5, 'buffer_distance': 1000}}


    # Based on analysis: Lithology and topographic features are most important
    
    def create_interaction_features(df):
        '''Create interaction features for better discrimination'''
        
        # Terrain roughness combinations
        df['slope_aspect_interaction'] = df['Slope_aligned'] * np.cos(np.radians(df['Aspect_aligned']))
        df['slope_curvature_combo'] = df['Slope_aligned'] * (df['planCurv_aligned'] + df['profileCurv_aligned'])
        
        # Topographic indices combinations
        df['tri_tpi_ratio'] = df['TRI_aligned'] / (df['TPI_aligned'] + 1e-6)
        df['twi_spi_combo'] = df['TWI_aligned'] * df['SPI_aligned']
        
        # Distance ratios (based on analysis showing proximity importance)
        df['river_road_ratio'] = df['distance_river_aligned'] / (df['distance_road_aligned'] + 1e-6)
        
        # Elevation-based features
        df['elevation_slope_interaction'] = df['dem_lo19_aligned'] * df['Slope_aligned']
        
        # Flow accumulation categories
        df['flow_acc_log'] = np.log1p(df['flowAcc_aligned'])
        
        return df
    
    def enhance_geological_features(df):
        '''Enhance geological features based on importance analysis'''
        
        # Create lithology diversity index
        lithology_cols = [col for col in df.columns if 'lithology_' in col]
        df['lithology_diversity'] = df[lithology_cols].sum(axis=1)
        
        # Create soil diversity index  
        soil_cols = [col for col in df.columns if 'soil_' in col]
        df['soil_diversity'] = df[soil_cols].sum(axis=1)
        
        # Geological stability indicator
        # Based on analysis: lithology_0.0 and lithology_2.0 are most important
        df['geological_stability'] = (df['lithology_0.0'] * 0.1091 + 
                                    df['lithology_2.0'] * 0.1037 +
                                    df['lithology_5.0'] * 0.1031)
        
        return df
    


    def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
        '''Calculate all relevant metrics for landslide prediction'''
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix,
                                   precision_recall_curve)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix analysis
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative_rate'] = tn / (tn + fp)
        metrics['false_positive_rate'] = fp / (fp + tn)
        metrics['landslide_capture_rate'] = tp / (tp + fn)
        
        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
        
        return metrics
    
    def spatial_autocorrelation_analysis(coordinates, residuals):
        '''Analyze spatial patterns in model residuals'''
        
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import spearmanr
        
        # Calculate spatial distances
        distances = pdist(coordinates)
        distance_matrix = squareform(distances)
        
        # Calculate Moran's I for spatial autocorrelation
        # Simplified version - in practice use proper spatial weights
        weights = 1.0 / (distance_matrix + 1e-6)
        np.fill_diagonal(weights, 0)
        
        n = len(residuals)
        mean_residual = np.mean(residuals)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * (residuals[i] - mean_residual) * (residuals[j] - mean_residual)
            denominator += (residuals[i] - mean_residual) ** 2
        
        moran_i = (n / np.sum(weights)) * (numerator / denominator)
        
        return moran_i
    