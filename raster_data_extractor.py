# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RasterDataExtractor
 Utility for extracting feature data from rasters using basic GDAL/rasterio
 Alternative to complex QGIS processing for simpler workflows
 ***************************************************************************/
"""

import os
import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Optional, Callable

try:
    import rasterio
    from rasterio.sample import sample_gen
    import geopandas as gpd
    from shapely.geometry import Point
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

class RasterDataExtractor:
    """Extract features from rasters using rasterio (fallback when QGIS unavailable)"""
    
    def __init__(self):
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio and geopandas are required. Install with: pip install rasterio geopandas")
    
    def extract_features_simple(self, raster_paths: List[str], landslide_points_path: str, 
                               generate_non_landslides: bool = True, 
                               progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Extract features from rasters at landslide points (simplified version)
        
        Args:
            raster_paths: List of paths to raster files
            landslide_points_path: Path to landslide points vector file  
            generate_non_landslides: Whether to generate non-landslide points
            progress_callback: Function to report progress (0-100)
            
        Returns:
            DataFrame with extracted features
        """
        
        if progress_callback:
            progress_callback(0)
            
        # Load landslide points
        try:
            gdf = gpd.read_file(landslide_points_path)
            landslide_coords = [(point.x, point.y) for point in gdf.geometry]
        except Exception as e:
            raise ValueError(f"Cannot load landslide points from {landslide_points_path}: {e}")
            
        if progress_callback:
            progress_callback(10)
            
        # Generate non-landslide points if requested
        non_landslide_coords = []
        if generate_non_landslides:
            # Use first raster to get bounds
            with rasterio.open(raster_paths[0]) as src:
                bounds = src.bounds
                
            non_landslide_coords = self._generate_random_points(
                bounds, landslide_coords, len(landslide_coords) * 2
            )
            
        if progress_callback:
            progress_callback(20)
            
        # Combine all coordinates with labels
        all_coords = []
        labels = []
        
        # Add landslide points (label = 1)
        for x, y in landslide_coords:
            all_coords.append((x, y))
            labels.append(1)
            
        # Add non-landslide points (label = 0)
        for x, y in non_landslide_coords:
            all_coords.append((x, y))
            labels.append(0)
            
        # Extract features from each raster
        feature_data = {'x': [c[0] for c in all_coords], 
                       'y': [c[1] for c in all_coords], 
                       'label': labels}
        
        total_rasters = len(raster_paths)
        
        for i, raster_path in enumerate(raster_paths):
            try:
                raster_name = os.path.basename(raster_path).split('.')[0]
                
                with rasterio.open(raster_path) as src:
                    # Sample all points at once
                    values = list(sample_gen(src, all_coords))
                    
                    # Extract first band values and handle NoData
                    raster_values = []
                    for val in values:
                        if val[0] is None or np.isnan(val[0]) or val[0] == src.nodata:
                            raster_values.append(0)
                        else:
                            raster_values.append(val[0])
                            
                    feature_data[raster_name] = raster_values
                    
            except Exception as e:
                print(f"Warning: Could not process raster {raster_path}: {e}")
                # Fill with zeros if raster can't be processed
                feature_data[raster_name] = [0] * len(all_coords)
                
            # Update progress
            if progress_callback:
                progress = 20 + int(((i + 1) / total_rasters) * 70)
                progress_callback(progress)
                
        if progress_callback:
            progress_callback(100)
            
        return pd.DataFrame(feature_data)
        
    def _generate_random_points(self, bounds: Tuple, existing_points: List[Tuple], 
                               num_points: int, min_distance: float = 100) -> List[Tuple]:
        """Generate random points within bounds, avoiding existing points"""
        
        random_points = []
        attempts = 0
        max_attempts = num_points * 10
        
        while len(random_points) < num_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point within bounds
            x = random.uniform(bounds.left, bounds.right)
            y = random.uniform(bounds.bottom, bounds.top)
            
            # Check minimum distance from existing points (simplified check)
            too_close = False
            for ex_x, ex_y in existing_points:
                distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                random_points.append((x, y))
                
        return random_points

def create_sample_data(output_dir: str = None) -> Tuple[str, str]:
    """
    Create sample CSV files for testing when raster extraction is not available
    Returns paths to landslide and non-landslide CSV files
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
        
    # Sample feature names (matching common raster types)
    features = [
        'Aspect', 'dem_lo19', 'flowAcc', 'planCurv', 'profileCurv',
        'distance_river', 'distance_road', 'Slope', 'SPI', 'TPI', 'TRI', 'TWI',
        'lithology_raster', 'soil_raster'
    ]
    
    # Generate sample landslide data
    np.random.seed(42)
    n_landslides = 100
    
    landslide_data = {}
    for feature in features:
        if 'distance' in feature.lower():
            # Distance features - closer to features for landslides
            landslide_data[feature] = np.random.exponential(scale=200, size=n_landslides)
        elif feature.lower() == 'slope':
            # Higher slopes for landslides
            landslide_data[feature] = np.random.normal(loc=25, scale=10, size=n_landslides)
        elif feature.lower() == 'aspect':
            # Random aspect
            landslide_data[feature] = np.random.uniform(0, 360, size=n_landslides)
        elif 'curv' in feature.lower():
            # Curvature values
            landslide_data[feature] = np.random.normal(loc=0, scale=0.1, size=n_landslides)
        else:
            # General features
            landslide_data[feature] = np.random.normal(loc=100, scale=50, size=n_landslides)
    
    # Add coordinates (dummy)
    landslide_data['xcoord'] = np.random.uniform(28.0, 31.0, n_landslides)
    landslide_data['ycoord'] = np.random.uniform(-30.0, -29.0, n_landslides)
    landslide_data['fid'] = range(n_landslides)
    
    # Generate sample non-landslide data
    n_non_landslides = 200
    
    non_landslide_data = {}
    for feature in features:
        if 'distance' in feature.lower():
            # Further from features for non-landslides
            non_landslide_data[feature] = np.random.exponential(scale=500, size=n_non_landslides)
        elif feature.lower() == 'slope':
            # Lower slopes for non-landslides
            non_landslide_data[feature] = np.random.normal(loc=10, scale=8, size=n_non_landslides)
        elif feature.lower() == 'aspect':
            # Random aspect
            non_landslide_data[feature] = np.random.uniform(0, 360, size=n_non_landslides)
        elif 'curv' in feature.lower():
            # Curvature values
            non_landslide_data[feature] = np.random.normal(loc=0, scale=0.05, size=n_non_landslides)
        else:
            # General features - different distribution
            non_landslide_data[feature] = np.random.normal(loc=80, scale=40, size=n_non_landslides)
    
    # Add coordinates (dummy)
    non_landslide_data['xcoord'] = np.random.uniform(28.0, 31.0, n_non_landslides)
    non_landslide_data['ycoord'] = np.random.uniform(-30.0, -29.0, n_non_landslides)
    non_landslide_data['fid'] = range(n_non_landslides)
    
    # Save to CSV files
    landslide_df = pd.DataFrame(landslide_data)
    non_landslide_df = pd.DataFrame(non_landslide_data)
    
    landslide_path = os.path.join(output_dir, 'output_landslides.csv')
    non_landslide_path = os.path.join(output_dir, 'output_non_landslides.csv')
    
    landslide_df.to_csv(landslide_path, index=False)
    non_landslide_df.to_csv(non_landslide_path, index=False)
    
    print(f"Sample data created:")
    print(f"  Landslides: {landslide_path}")
    print(f"  Non-landslides: {non_landslide_path}")
    
    return landslide_path, non_landslide_path

if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data()
