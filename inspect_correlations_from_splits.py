#!/usr/bin/env python3
"""
Script to inspect correlations between sensors and target variable across data splits.

This script:
1. Loads all possible sensor names from metadata.parquet
2. For each data split, calculates Spearman correlation with target variable
3. Exports correlation results for each split
4. Pools results to calculate mean correlations across all splits
"""

import os
import glob
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import argparse


def load_all_sensor_ids(metadata_path):
    """Load all possible sensor IDs from metadata.parquet."""
    print("Loading sensor metadata...")
    metadata_df = pd.read_parquet(metadata_path)
    sensor_ids = metadata_df['object_id'].unique().tolist()
    print(f"Found {len(sensor_ids)} unique sensors in metadata")
    return sensor_ids


def get_data_splits(splits_dir):
    """Get all available data split directories."""
    split_dirs = [d for d in os.listdir(splits_dir) 
                  if os.path.isdir(os.path.join(splits_dir, d)) and d.startswith('train_')]
    print(f"Found {len(split_dirs)} data splits")
    return sorted(split_dirs)


def calculate_correlations_for_split(split_dir, all_sensor_ids, target_variable, target_shift_minutes=0):
    """Calculate Spearman correlations for a single data split.
    
    Args:
        split_dir: Path to the data split directory
        all_sensor_ids: List of all sensor IDs to analyze
        target_variable: Target variable name
        target_shift_minutes: Minutes to shift target data forward (positive) or backward (negative)
    """
    train_file = os.path.join(split_dir, "preproc_full_train_df.parquet")
    
    if not os.path.exists(train_file):
        print(f"Warning: Training file not found: {train_file}")
        return None
    
    print(f"Processing split: {os.path.basename(split_dir)}")
    
    # Load training data
    train_df = pd.read_parquet(train_file)
    available_sensors = set(train_df.columns)
    
    print(f"  Available sensors in split: {len(available_sensors)}")
    
    # Check if target variable exists
    if target_variable not in available_sensors:
        print(f"  Warning: Target variable {target_variable} not found in split")
        return None
    
    # Get target variable values and apply time shift if specified
    target_data = train_df[[target_variable]].dropna()
    
    if len(target_data) == 0:
        print(f"  Warning: No valid target values in split")
        return None
    
    # Apply time shift to target data if specified
    if target_shift_minutes != 0:
        target_data = target_data.copy()
        target_data.index = target_data.index + pd.Timedelta(minutes=target_shift_minutes)
        print(f"  Applied {target_shift_minutes} minute shift to target variable")
    
    target_values = target_data[target_variable]
    
    correlations = []
    
    # Calculate correlations for all sensors
    for sensor_id in tqdm(all_sensor_ids, desc="  Computing correlations"):
        if sensor_id == target_variable:
            # Perfect correlation with itself
            corr = 1.0
        elif sensor_id in available_sensors:
            # Sensor exists in this split
            sensor_values = train_df[sensor_id].dropna()
            
            if len(sensor_values) == 0:
                corr = 0.0
            else:
                # Align indices for correlation calculation (after potential time shift)
                common_idx = target_values.index.intersection(sensor_values.index)
                if len(common_idx) < 10:  # Need at least 10 points for meaningful correlation
                    corr = 0.0
                else:
                    try:
                        corr, _ = spearmanr(target_values.loc[common_idx], 
                                          sensor_values.loc[common_idx])
                        if np.isnan(corr):
                            corr = 0.0
                    except:
                        corr = 0.0
        else:
            # Sensor not available in this split
            corr = 0.0
        
        correlations.append({
            'sensor_id': sensor_id,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(correlations)
    
    # Add split information
    results_df['data_split'] = os.path.basename(split_dir)
    
    # Calculate ranks based on absolute correlation (1 = highest absolute correlation)
    results_df['abs_correlation_rank'] = results_df['abs_correlation'].rank(method='dense', ascending=False)
    
    print(f"  Computed correlations for {len(results_df)} sensors")
    print(f"  Non-zero correlations: {(results_df['abs_correlation'] > 0).sum()}")
    
    return results_df


def export_split_correlations(results_df, output_dir, split_name):
    """Export correlation results for a single split."""
    output_file = os.path.join(output_dir, f"correlations_{split_name}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"  Exported: {output_file}")


def pool_and_summarize_correlations(all_results, output_dir):
    """Pool correlation results from all splits and calculate summary statistics."""
    print("\nPooling results from all splits...")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"Total correlation records: {len(combined_df)}")
    print(f"Unique sensors: {combined_df['sensor_id'].nunique()}")
    print(f"Data splits: {combined_df['data_split'].nunique()}")
    
    # Calculate mean correlations and rank statistics per sensor
    summary_stats = combined_df.groupby('sensor_id').agg({
        'correlation': ['mean', 'std', 'count'],
        'abs_correlation': ['mean', 'std', 'max', 'min'],
        'abs_correlation_rank': ['mean', 'std', 'max', 'min']
    }).round(6)
    
    # Flatten column names
    summary_stats.columns = [f"{col[1]}_{col[0]}" if col[1] != '' else col[0] 
                            for col in summary_stats.columns]
    summary_stats = summary_stats.rename(columns={
        'mean_correlation': 'mean_correlation',
        'std_correlation': 'std_correlation', 
        'count_correlation': 'split_count',
        'mean_abs_correlation': 'mean_abs_correlation',
        'std_abs_correlation': 'std_abs_correlation',
        'max_abs_correlation': 'max_abs_correlation',
        'min_abs_correlation': 'min_abs_correlation',
        'mean_abs_correlation_rank': 'mean_rank',
        'std_abs_correlation_rank': 'std_rank',
        'max_abs_correlation_rank': 'max_rank',
        'min_abs_correlation_rank': 'min_rank'
    })
    
    # Reset index to make sensor_id a column
    summary_stats = summary_stats.reset_index()
    
    # Sort by mean rank (ascending - lower rank means higher correlation)
    summary_stats = summary_stats.sort_values('mean_rank', ascending=True)
    
    # Export summary
    summary_file = os.path.join(output_dir, "correlation_summary_across_splits.csv")
    summary_stats.to_csv(summary_file, index=False)
    
    # Export detailed results
    detailed_file = os.path.join(output_dir, "all_correlations_by_split.csv") 
    combined_df.to_csv(detailed_file, index=False)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total unique sensors analyzed: {len(summary_stats)}")
    print(f"Sensors with non-zero correlations: {(summary_stats['mean_abs_correlation'] > 0).sum()}")
    print(f"Sensors present in all splits: {(summary_stats['split_count'] == combined_df['data_split'].nunique()).sum()}")
    
    print(f"\n🔝 TOP 10 SENSORS BY MEAN RANK (BASED ON ABSOLUTE CORRELATION):")
    top_sensors = summary_stats.head(10)[['sensor_id', 'mean_correlation', 'mean_abs_correlation', 'mean_rank', 'std_rank', 'split_count']]
    print(top_sensors.to_string(index=False))
    
    print(f"\n📊 RANK STATISTICS SUMMARY:")
    print(f"Best possible rank (highest correlation): 1.0")
    print(f"Worst possible rank (lowest correlation): {combined_df['sensor_id'].nunique()}")
    print(f"Average rank across all sensors: {summary_stats['mean_rank'].mean():.2f}")
    print(f"Sensors with mean rank ≤ 10: {(summary_stats['mean_rank'] <= 10).sum()}")
    print(f"Sensors with mean rank ≤ 50: {(summary_stats['mean_rank'] <= 50).sum()}")
    print(f"Sensors with mean rank ≤ 100: {(summary_stats['mean_rank'] <= 100).sum()}")
    
    print(f"\n💾 FILES EXPORTED:")
    print(f"  Summary statistics: {summary_file}")
    print(f"  Detailed results: {detailed_file}")
    
    return summary_stats, combined_df


def main():
    parser = argparse.ArgumentParser(description='Inspect correlations across data splits')
    parser.add_argument('--metadata_path', type=str, 
                       default='data/kaggle_dl/metadata.parquet',
                       help='Path to metadata.parquet file')
    parser.add_argument('--splits_dir', type=str,
                       default='data/splits', 
                       help='Directory containing data splits')
    parser.add_argument('--target_variable', type=str,
                       default='B205WC000.AM02',
                       help='Target variable name')
    parser.add_argument('--output_dir', type=str,
                       default='correlation_analysis',
                       help='Output directory for results')
    parser.add_argument('--target_shift_minutes', type=int, default=0,
                       help='Shift target variable by specified minutes to measure correlation with future values (default: 0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 CORRELATION ANALYSIS ACROSS DATA SPLITS")
    print("=" * 50)
    print(f"Target variable: {args.target_variable}")
    print(f"Target shift: {args.target_shift_minutes} minutes")
    print("-" * 50)
    
    # Step 1: Load all sensor IDs
    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file not found: {args.metadata_path}")
        return
    
    all_sensor_ids = load_all_sensor_ids(args.metadata_path)
    
    # Step 2: Get data splits
    if not os.path.exists(args.splits_dir):
        print(f"Error: Splits directory not found: {args.splits_dir}")
        return
    
    data_splits = get_data_splits(args.splits_dir)
    
    if not data_splits:
        print("No data splits found!")
        return
    
    print(f"Data splits to process: {data_splits}")
    
    # Step 3: Process each split
    all_results = []
    
    for split_name in data_splits:
        split_path = os.path.join(args.splits_dir, split_name)
        
        # Calculate correlations for this split
        split_results = calculate_correlations_for_split(
            split_path, all_sensor_ids, args.target_variable, args.target_shift_minutes
        )
        
        if split_results is not None:
            # Export individual split results
            export_split_correlations(split_results, args.output_dir, split_name)
            all_results.append(split_results)
        else:
            print(f"  Skipping split {split_name} due to errors")
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Step 4: Pool and summarize results
    summary_stats, combined_df = pool_and_summarize_correlations(all_results, args.output_dir)
    
    print(f"\n✅ Analysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()