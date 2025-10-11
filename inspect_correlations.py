#!/usr/bin/env python3
"""
Inspect correlations between sensors across multiple buildings and time periods.

This script loads sensor data from specified buildings and measures the 
Spearman correlation of resampled time series data with a specified target sensor.
Supports analysis across multiple months and buildings.
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

def validate_and_parse_month(month_str: str) -> str:
    """
    Validate and parse month string in YYYY-MM format.
    
    Args:
        month_str: Month string in YYYY-MM format
        
    Returns:
        str: Validated month string in YYYY-MM format
        
    Raises:
        ValueError: If format is invalid or date is out of supported range
    """
    try:
        # Parse the date to validate format
        parsed_date = datetime.strptime(month_str, '%Y-%m')
        
        # Check if it's within the supported range (2025-01 to 2025-07)
        if parsed_date > datetime(2025, 7, 31):
            raise ValueError(f"Month {month_str} is outside supported range (up to 2025-07)")
            
        return month_str
    except ValueError as e:
        if "time data" in str(e):
            raise ValueError(f"Invalid month format: {month_str}. Expected format: YYYY-MM")
        else:
            raise e

def generate_month_range(start_month: str, end_month: str) -> List[str]:
    """
    Generate list of month strings from start to end month.
    
    Args:
        start_month: Start month in YYYY-MM format
        end_month: End month in YYYY-MM format
        
    Returns:
        List[str]: List of month strings in RBHU-YYYY-MM format
    """
    start_date = datetime.strptime(start_month, '%Y-%m')
    end_date = datetime.strptime(end_month, '%Y-%m')
    
    if start_date > end_date:
        raise ValueError(f"Start month ({start_month}) cannot be after end month ({end_month})")
    
    months = []
    current_date = start_date
    
    while current_date <= end_date:
        # Convert to RBHU format (RBHU-YYYY-MM)
        month_str = f"RBHU-{current_date.strftime('%Y-%m')}"
        months.append(month_str)
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return months

def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load the metadata parquet file."""
    metadata_path = os.path.join(data_dir, "metadata.parquet")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_parquet(metadata_path)

def get_building_sensors(metadata: pd.DataFrame, buildings: List[str]) -> List[str]:
    """Get all sensor IDs for the specified buildings."""
    building_metadata = metadata[metadata['building'].isin(buildings)]
    return building_metadata['object_id'].tolist()

def explore_building_structure(data_dir: str, month: str, building: str) -> None:
    """
    Debug function to explore and print the folder structure of a building.
    
    Args:
        data_dir: Base data directory
        month: Month string (e.g., 'RBHU-2025-01')
        building: Building identifier (e.g., 'B201', 'B205', 'B106')
    """
    building_base_path = os.path.join(data_dir, month, 'RBHU', building)
    
    if not os.path.exists(building_base_path):
        print(f"Building path does not exist: {building_base_path}")
        return
    
    print(f"Exploring structure for {building} in {month}:")
    
    # Walk through all subdirectories and list parquet files
    for root, dirs, files in os.walk(building_base_path):
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if parquet_files:
            relative_path = os.path.relpath(root, building_base_path)
            if relative_path == '.':
                print(f"  Direct files: {len(parquet_files)} parquet files")
            else:
                print(f"  Subfolder '{relative_path}': {len(parquet_files)} parquet files")

def find_sensor_files(data_dir: str, month: str, building: str, sensor_id: str) -> List[str]:
    """
    Find all instances of a sensor file across different folder structures.
    
    Args:
        data_dir: Base data directory
        month: Month string (e.g., 'RBHU-2025-01')
        building: Building identifier (e.g., 'B201', 'B205', 'B106')
        sensor_id: Sensor ID to search for
        
    Returns:
        List of paths to matching sensor files
    """
    sensor_files = []
    building_base_path = os.path.join(data_dir, month, 'RBHU', building)
    
    if not os.path.exists(building_base_path):
        return sensor_files
    
    # Use glob to recursively search for the sensor file
    search_pattern = os.path.join(building_base_path, '**', f'{sensor_id}.parquet')
    matching_files = glob.glob(search_pattern, recursive=True)
    
    # Also check direct path (for buildings like B106, B205)
    direct_path = os.path.join(building_base_path, f'{sensor_id}.parquet')
    if os.path.exists(direct_path) and direct_path not in matching_files:
        matching_files.append(direct_path)
    
    return matching_files

def load_sensor_data(data_dir: str, buildings: List[str], sensor_id: str, start_month: str, end_month: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Load time series data for a specific sensor from multiple buildings and months."""
    dataframes = []
    
    # Validate and generate month strings
    validate_and_parse_month(start_month)
    validate_and_parse_month(end_month)
    months = generate_month_range(start_month, end_month)
    
    total_files_found = 0
    total_files_loaded = 0
    
    for month in months:
        for building in buildings:
            # Find all instances of the sensor file in the building folder structure
            sensor_files = find_sensor_files(data_dir, month, building, sensor_id)
            total_files_found += len(sensor_files)
            
            if verbose and sensor_files:
                print(f"Found {len(sensor_files)} files for {sensor_id} in {building}/{month}")
            
            for sensor_path in sensor_files:
                try:
                    df = pd.read_parquet(sensor_path)
                    # Ensure the dataframe has the expected columns
                    if 'time' not in df.columns or 'data' not in df.columns:
                        print(f"Warning: Expected columns 'time' and 'data' not found in {sensor_id} at {sensor_path}")
                        continue
                    
                    # Check for empty dataframe
                    if df.empty:
                        print(f"Warning: Empty dataframe for {sensor_id} at {sensor_path}")
                        continue
                        
                    dataframes.append(df)
                    total_files_loaded += 1
                    
                    if verbose:
                        print(f"Successfully loaded {sensor_id} from {sensor_path} ({len(df)} rows)")
                        
                except Exception as e:
                    print(f"Error loading {sensor_id} from {sensor_path}: {e}")
                    continue
    
    if verbose:
        print(f"Summary for {sensor_id}: Found {total_files_found} files, successfully loaded {total_files_loaded}")
    
    if not dataframes:
        print(f"Warning: No sensor data found for {sensor_id} in any of the specified buildings and months")
        return None
    
    # Concatenate all dataframes and sort by time
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    # Remove duplicates if any (in case the same file was found multiple times)
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['time']).reset_index(drop=True)
    final_rows = len(combined_df)
    
    if verbose and initial_rows != final_rows:
        print(f"Removed {initial_rows - final_rows} duplicate rows for {sensor_id}")
    
    return combined_df

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

def analyze_correlations(data_dir: str, buildings: List[str], target_sensor: str, start_month: str, end_month: str, resample_freq: str = '1min', target_shift_minutes: int = 0, verbose: bool = False) -> Dict[str, float]:
    """
    Analyze correlations between target sensor and all other sensors in the specified buildings.
    
    Args:
        data_dir: Path to the data directory
        buildings: List of building identifiers (e.g., ['B205', 'B106'])
        target_sensor: Target sensor ID (e.g., 'B205WC000.AM02')
        start_month: Start month for analysis (format: YYYY-MM)
        end_month: End month for analysis (format: YYYY-MM)
        resample_freq: Resampling frequency (default: '1min')
        target_shift_minutes: Minutes to shift target data for future correlation analysis (default: 0)
    
    Returns:
        Dictionary mapping sensor IDs to their correlation with the target sensor
    """
    print(f"Analyzing correlations for buildings: {', '.join(buildings)}")
    print(f"Target sensor: {target_sensor}")
    print(f"Month range: {start_month} to {end_month}")
    print(f"Resampling frequency: {resample_freq}")
    print(f"Target shift: {target_shift_minutes} minutes")
    print("-" * 50)
    
    # Validate month range (validation will be done in load_sensor_data)
    validate_and_parse_month(start_month)
    validate_and_parse_month(end_month)
    
    # Load metadata
    metadata = load_metadata(data_dir)
    
    # Get all sensors for the buildings
    building_sensors = get_building_sensors(metadata, buildings)
    print(f"Found {len(building_sensors)} sensors across {len(buildings)} buildings")
    
    # Load and resample target sensor data
    target_data = load_sensor_data(data_dir, buildings, target_sensor, start_month, end_month, verbose)
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
        
        sensor_data = load_sensor_data(data_dir, buildings, sensor_id, start_month, end_month, verbose)
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
    parser = argparse.ArgumentParser(description='Analyze sensor correlations across multiple buildings and time periods')
    parser.add_argument('--data-dir', default='data/kaggle_dl', 
                       help='Path to the data directory (default: data/kaggle_dl)')
    parser.add_argument('--buildings', nargs='+', required=True,
                       help='Building identifiers (e.g., B205 B106) - supports multiple buildings')
    parser.add_argument('--target-sensor', required=True,
                       help='Target sensor ID (e.g., B205WC000.AM02)')
    parser.add_argument('--start-month', type=str, default='2025-01',
                       help='Start month for analysis (format: YYYY-MM, default: 2025-01)')
    parser.add_argument('--end-month', type=str, default='2025-01',
                       help='End month for analysis (format: YYYY-MM, default: 2025-01)')
    parser.add_argument('--resample-freq', default='1min',
                       help='Resampling frequency (default: 1min)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top correlations to display (default: 10)')
    parser.add_argument('--target-shift-minutes', type=int, default=0,
                       help='Shift target variable by specified minutes to measure correlation with future values (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file to save results (optional)')
    parser.add_argument('--debug-structure', action='store_true',
                       help='Enable debug mode to explore building folder structures')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for detailed loading information')
    
    args = parser.parse_args()
    
    try:
        # If debug mode is enabled, explore building structures first
        if args.debug_structure:
            print("=== DEBUG: Exploring building structures ===")
            validate_and_parse_month(args.start_month)
            months = generate_month_range(args.start_month, args.start_month)  # Just check first month
            for month in months[:1]:  # Only check first month for debugging
                for building in args.buildings:
                    explore_building_structure(args.data_dir, month, building)
                    print()
            print("=== End debug exploration ===\n")
        
        # Analyze correlations
        correlations = analyze_correlations(
            args.data_dir, 
            args.buildings, 
            args.target_sensor,
            args.start_month,
            args.end_month,
            args.resample_freq,
            args.target_shift_minutes,
            args.verbose
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