#!/usr/bin/env python3
"""
Inspect correlations between sensors in a building.

This script loads sensor data from a specified building and measures the 
Spearman correlation of resampled time series data with a specified target sensor.
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import warnings

def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load the metadata parquet file."""
    metadata_path = os.path.join(data_dir, "metadata.parquet")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_parquet(metadata_path)

def get_building_sensors(metadata: pd.DataFrame, building: str) -> List[str]:
    """Get all sensor IDs for a specific building."""
    building_metadata = metadata[metadata['building'] == building]
    return building_metadata['object_id'].tolist()

def load_sensor_data(data_dir: str, building: str, sensor_id: str) -> Optional[pd.DataFrame]:
    """Load time series data for a specific sensor."""
    # Construct the path based on the structure seen in FirstLook.ipynb
    sensor_path = os.path.join(data_dir, 'RBHU-2025-01', 'RBHU', building, f'{sensor_id}.parquet')
    
    if not os.path.exists(sensor_path):
        print(f"Warning: Sensor data not found for {sensor_id} at {sensor_path}")
        return None
    
    try:
        df = pd.read_parquet(sensor_path)
        # Ensure the dataframe has the expected columns
        if 'time' not in df.columns or 'data' not in df.columns:
            print(f"Warning: Expected columns 'time' and 'data' not found in {sensor_id}")
            return None
        return df
    except Exception as e:
        print(f"Error loading {sensor_id}: {e}")
        return None

def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample time series data to specified frequency with interpolation."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Resample following the pattern from FirstLook.ipynb
    resampled = df.set_index('time').resample(freq).mean().interpolate().reset_index()
    return resampled

def calculate_spearman_correlation(target_data: pd.DataFrame, sensor_data: pd.DataFrame, target_shift_minutes: int = 0) -> Optional[float]:
    """
    Calculate Spearman correlation between target and sensor data.
    
    Args:
        target_data: DataFrame with target sensor data
        sensor_data: DataFrame with sensor data  
        target_shift_minutes: Minutes to shift target data forward (positive) or backward (negative)
    """
    if target_data.empty or sensor_data.empty:
        return None
    
    # Apply time shift to target data if specified
    target_shifted = target_data.copy()
    if target_shift_minutes != 0:
        target_shifted['time'] = target_shifted['time'] + pd.Timedelta(minutes=target_shift_minutes)
    
    # Merge on time to ensure alignment
    merged = pd.merge(target_shifted, sensor_data, on='time', suffixes=('_target', '_sensor'))
    
    if merged.empty or len(merged) < 2:
        return None
    
    # Remove NaN values
    merged = merged.dropna()
    
    if len(merged) < 2:
        return None
    
    try:
        correlation, p_value = spearmanr(merged['data_target'], merged['data_sensor'])
        return correlation if not np.isnan(correlation) else None
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return None

def analyze_correlations(data_dir: str, building: str, target_sensor: str, resample_freq: str = '1min', target_shift_minutes: int = 0) -> Dict[str, float]:
    """
    Analyze correlations between target sensor and all other sensors in the building.
    
    Args:
        data_dir: Path to the data directory
        building: Building identifier (e.g., 'B205')
        target_sensor: Target sensor ID (e.g., 'B205WC000.AM02')
        resample_freq: Resampling frequency (default: '1min')
        target_shift_minutes: Minutes to shift target data for future correlation analysis (default: 0)
    
    Returns:
        Dictionary mapping sensor IDs to their correlation with the target sensor
    """
    print(f"Analyzing correlations for building {building}")
    print(f"Target sensor: {target_sensor}")
    print(f"Resampling frequency: {resample_freq}")
    print(f"Target shift: {target_shift_minutes} minutes")
    print("-" * 50)
    
    # Load metadata
    metadata = load_metadata(data_dir)
    
    # Get all sensors for the building
    building_sensors = get_building_sensors(metadata, building)
    print(f"Found {len(building_sensors)} sensors in building {building}")
    
    # Load and resample target sensor data
    target_data = load_sensor_data(data_dir, building, target_sensor)
    if target_data is None:
        raise ValueError(f"Could not load target sensor data: {target_sensor}")
    
    target_resampled = resample_data(target_data, resample_freq)
    print(f"Target sensor data shape after resampling: {target_resampled.shape}")
    
    # Calculate correlations
    correlations = {}
    processed_count = 0
    
    for sensor_id in building_sensors:
        if sensor_id == target_sensor:
            continue  # Skip self-correlation
        
        sensor_data = load_sensor_data(data_dir, building, sensor_id)
        if sensor_data is None:
            continue
        
        sensor_resampled = resample_data(sensor_data, resample_freq)
        correlation = calculate_spearman_correlation(target_resampled, sensor_resampled, target_shift_minutes)
        
        if correlation is not None:
            correlations[sensor_id] = correlation
            processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} sensors...")
    
    print(f"Successfully calculated correlations for {len(correlations)} sensors")
    return correlations

def print_results(correlations: Dict[str, float], top_n: int = 10):
    """Print correlation results sorted by absolute correlation value."""
    if not correlations:
        print("No correlations calculated.")
        return
    
    # Sort by absolute correlation value
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {top_n} correlations (by absolute value):")
    print("-" * 60)
    print(f"{'Sensor ID':<20} {'Correlation':<15} {'Abs Correlation':<15}")
    print("-" * 60)
    
    for i, (sensor_id, correlation) in enumerate(sorted_correlations[:top_n]):
        print(f"{sensor_id:<20} {correlation:>8.4f}      {abs(correlation):>8.4f}")
    
    print(f"\nSummary statistics:")
    correlations_values = list(correlations.values())
    print(f"Mean correlation: {np.mean(correlations_values):.4f}")
    print(f"Std correlation: {np.std(correlations_values):.4f}")
    print(f"Max correlation: {np.max(correlations_values):.4f}")
    print(f"Min correlation: {np.min(correlations_values):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze sensor correlations in a building')
    parser.add_argument('--data-dir', default='data/kaggle_dl', 
                       help='Path to the data directory (default: data/kaggle_dl)')
    parser.add_argument('--building', required=True,
                       help='Building identifier (e.g., B205)')
    parser.add_argument('--target-sensor', required=True,
                       help='Target sensor ID (e.g., B205WC000.AM02)')
    parser.add_argument('--resample-freq', default='1min',
                       help='Resampling frequency (default: 1min)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top correlations to display (default: 10)')
    parser.add_argument('--target-shift-minutes', type=int, default=0,
                       help='Shift target variable by specified minutes to measure correlation with future values (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file to save results (optional)')
    
    args = parser.parse_args()
    
    try:
        # Analyze correlations
        correlations = analyze_correlations(
            args.data_dir, 
            args.building, 
            args.target_sensor, 
            args.resample_freq,
            args.target_shift_minutes
        )

        # Print results
        print_results(correlations, args.top_n)
        
        # Save to CSV if requested
        if args.output:
            results_df = pd.DataFrame([
                {'sensor_id': sensor_id, 'correlation': correlation, 'abs_correlation': abs(correlation)}
                for sensor_id, correlation in correlations.items()
            ]).sort_values('abs_correlation', ascending=False)
            
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())