"""Feature Selection Script for ELIAS Bosch AI Building Optimization Competition

This script provides a simplified approach using sklearn LASSO for feature selection and prediction.
It allows configurable training/testing months and exports feature weights for analysis.

Key features:
- Configurable training and testing months (YYYY-MM format)
- LASSO regression instead of neural networks
- Feature weight export to CSV
- Option to use all available features
"""

# Copyright (c) 2025 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
from datetime import datetime
import glob
from itertools import chain
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

EXAMPLE_PREDICTOR_VARIABLE_NAMES = [
    #ORIGINAL 2:
    "B205WC000.AM01",  # a supply temperature chilled water
    "B106WS01.AM54",  # an external temperature
    #best setup so far
    'B205WC140.AC21',# PRIMARY VALVE 1
    'B205HW010.PA11',# NUMBER OF STARTS
    'B205HW020.PA11',# NUMBER OF STARTS
    'B205WC001.AM71',# TOTAL VOLUME CHILLED WATER
    'B205WC000.AM71',# VOLUME CHILLED WATER BP201/202/206
    'B205HP110.AM55_3',# ACTUAL CAPACITY
    'B205WC030.AC63',# SETPOINT CHILLED WATER PUMP
    'B205WC030.AM51_4',# RUN ENABLED
    'B205WC030.AM53_1',# EVAPORATOR FLOW SWITCH STATUS
    'B205WC002.RA001',# SPEED CHILLED WATER PUMP
    'B205HW000.PA72',# VOLUME FEEDING HOT WATER SYSTEM
    'B205HW020.AC62',# SPEED SECONDARY PUMP
    'B205WC010.AM51_4',# RUN ENABLED
    'B205WC010.AM51_3',# CHILLER STATE
    'B205WC100.DM091_1',# HEAT PUMP HP110 READY - HEATING
    'B205WC000.DM90',# MAX. TEMP. CHILLED WATER
]


def generate_file_paths_from_months(data_dir: str, start_month: str, end_month: str) -> list:
    """
    Generate file paths for the specified month range.
    
    Args:
        data_dir: Base data directory
        start_month: Start month in YYYY-MM format
        end_month: End month in YYYY-MM format
        
    Returns:
        List of file paths
    """
    file_paths = []
    
    # Parse start and end dates
    start_date = datetime.strptime(start_month, '%Y-%m')
    end_date = datetime.strptime(end_month, '%Y-%m')
    
    current_date = start_date
    while current_date <= end_date:
        month_str = f"RBHU-{current_date.strftime('%Y-%m')}"
        pattern = f"{data_dir}/kaggle_dl/{month_str}/RBHU/**/*.parquet"
        month_files = glob.glob(pattern, recursive=True)
        file_paths.extend(month_files)
        print(f"Found {len(month_files)} files for {month_str}")
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return file_paths


def simple_load_and_resample_data(
    data_file_paths, generate_sample_plots=None, save_load_df=None, resample_freq_min=10, outputs_dir="outputs"
):
    """Load measurement timeseries data from the original data folder structure, turn the original irregularly sampling
    into a regular, common timeline, and join everything into one multivariate timeseries dataframe.

    Args:
        data_file_paths: List of file paths to the files containing the monthly measurement data for each sensor,
            should be in the right monthly order and overall original structure.
        generate_sample_plots: If provided, a list of sensor names to plot the timeseries of, and save the plots.
        save_load_df: If provided, the path to save the preprocessed data as a parquet file, and load it from there
            if it already exists.

    Returns:
        A pandas DataFrame containing the multivariate timeseries data, with the columns being the sensors,
        and the index being the datetime.
    """

    save = True
    if save_load_df and os.path.exists(save_load_df):
        print(f"Loading preprocessed data from {save_load_df} ...")
        multivariate_timeseries_df = pd.read_parquet(save_load_df)
        save = False

    else:
        # load, preprocess and group timeseries per sensor (i.e., for each sensor, got multiple periods):
        print("Start loading and preprocessing a dataset ...")
        dataframes_per_sensor = {}
        for path in tqdm(data_file_paths):
            name = path.split("/")[-1].replace(".parquet", "")
            if not name in dataframes_per_sensor.keys():
                dataframes_per_sensor[name] = []
            df_orig = pd.read_parquet(path, columns=["time", "data"])
            df_orig = df_orig.rename(columns={"data": name})
            df_orig["time"] = pd.to_datetime(df_orig["time"])
            df_orig = df_orig.set_index("time", drop=True)
            dataframes_per_sensor[name].append(df_orig)

        # join them along time dimension into one timeseries per sensor, and interpolate to regular frequency sample:
        print("Start joining data ...")
        regular_dataframe_per_sensor = {}
        for name, dfs in tqdm(dataframes_per_sensor.items()):
            # the above forward fill is justified by the fact that prior knowledge indicates, that measurement recordings
            # seem to be change-tirggered:
            resample_freq_str = f"{str(resample_freq_min)}min"
            regular_dataframe_per_sensor[name] = (
                pd.concat(dfs, axis=0)
                .resample(resample_freq_str)
                .ffill()  # interpolate via forward fill
            )

        # join into one multivariate timeseries dataframe, i.e., along column axis:
        multivariate_timeseries_df = pd.concat(
            regular_dataframe_per_sensor.values(), join="outer", axis=1
        ).ffill()  # forward fill again, due to different ends of concatenants otherwise leading to NaNs

    if generate_sample_plots:
        n_plots = len(generate_sample_plots)
        fig, axs = plt.subplots(
            n_plots,
            1,
            sharex=True,
            dpi=500,
        )
        plt.title("Input data timeseries")
        for i, col in enumerate(generate_sample_plots):
            if col in multivariate_timeseries_df.columns:
                axs[i].plot(multivariate_timeseries_df[col], label=col, linewidth=0.75)
                axs[i].tick_params(axis="x", labelrotation=90)
                axs[i].legend(fontsize="small")
        plt.savefig(f"{outputs_dir}/input_data_sample_timeseries_plot.png")
        plt.close(fig)

    if save_load_df and save:
        multivariate_timeseries_df.to_parquet(save_load_df)

    print("Done.")

    return multivariate_timeseries_df


def prepare_features_and_target(df, target_variable, predictor_variables=None, all_features=False):
    """
    Prepare feature matrix and target vector for LASSO regression.
    
    Args:
        df: Input dataframe with time series data
        target_variable: Name of target variable
        predictor_variables: List of predictor variable names (if None and all_features=True, use all available)
        all_features: If True, use all available columns except target as features
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        valid_indices: Boolean mask of valid (non-NaN) samples
    """
    
    if all_features:
        # Use all available columns except the target variable
        available_columns = [col for col in df.columns if col != target_variable]
        print(f"Using all available features: {len(available_columns)} features")
        feature_names = available_columns
    else:
        # Use specified predictor variables, filter out missing ones
        if predictor_variables is None:
            predictor_variables = EXAMPLE_PREDICTOR_VARIABLE_NAMES
            
        available_predictors = [col for col in predictor_variables if col in df.columns]
        missing_predictors = [col for col in predictor_variables if col not in df.columns]
        
        if missing_predictors:
            print(f"Warning: Missing predictor variables: {missing_predictors}")
        
        print(f"Using {len(available_predictors)} out of {len(predictor_variables)} specified features")
        feature_names = available_predictors
    
    # Check if target variable exists
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataframe columns")
    
    # Extract features and target
    X = df[feature_names].values
    y = df[target_variable].values
    
    # Find valid samples (no NaN values in features or target)
    valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    
    print(f"Valid samples: {valid_indices.sum()} out of {len(df)} ({valid_indices.sum()/len(df)*100:.1f}%)")
    
    return X[valid_indices], y[valid_indices], feature_names, valid_indices


def prepare_test_features_and_target(df, target_variable, training_feature_names):
    """
    Prepare test features ensuring consistency with training features.
    
    Args:
        df: Test dataframe
        target_variable: Name of target variable
        training_feature_names: List of feature names from training (in correct order)
        
    Returns:
        X: Feature matrix with same features as training (missing features filled with zeros)
        y: Target vector
        valid_indices: Boolean mask of valid (non-NaN) samples for target
    """
    
    # Check if target variable exists
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in test dataframe columns")
    
    # Create feature matrix with training feature order
    X_dict = {}
    
    # Fill available features
    for feature in training_feature_names:
        if feature in df.columns:
            X_dict[feature] = df[feature].values
        else:
            print(f"Warning: Feature '{feature}' missing in test data, filling with zeros")
            X_dict[feature] = np.zeros(len(df))
    
    # Create feature matrix in correct order
    X = np.column_stack([X_dict[feature] for feature in training_feature_names])
    y = df[target_variable].values
    
    # Find valid samples (no NaN values in target)
    # For features, we handle NaN by replacing with 0 (since missing features are already 0)
    X_nan_mask = np.isnan(X)
    X[X_nan_mask] = 0  # Replace NaN values in features with 0
    
    valid_indices = ~np.isnan(y)  # Only check target for validity
    
    print(f"Test set aligned: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Valid samples (non-NaN target): {valid_indices.sum()} out of {len(df)} ({valid_indices.sum()/len(df)*100:.1f}%)")
    if X_nan_mask.any():
        nan_features = [training_feature_names[i] for i in range(len(training_feature_names)) if X_nan_mask[:, i].any()]
        print(f"Warning: Number of NaN values in features filled with zeros: {len(nan_features)}")
        #print(f"Features with NaN values filled with zeros: {nan_features}")
    
    return X[valid_indices], y[valid_indices], valid_indices


def train_lasso_model(X_train, y_train, alpha=None, cv_folds=5, alpha_candidates=None):
    """
    Train LASSO regression model with grid search cross-validation for alpha selection.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization parameter (if None, use grid search cross-validation)
        cv_folds: Number of cross-validation folds for alpha selection
        alpha_candidates: List of alpha values to test (if None, use default range)
        
    Returns:
        Fitted LASSO model
        Selected alpha value
    """
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if alpha is None:
        print(f"Performing {cv_folds}-fold grid search cross-validation to select optimal alpha...")
        
        # Define alpha candidates with different magnitudes
        if alpha_candidates is None:
            alpha_candidates = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        
        print(f"Testing alpha values: {alpha_candidates}")
        
        # Create LASSO model for grid search
        lasso_base = Lasso(random_state=42, max_iter=2000)
        
        # Set up parameter grid
        param_grid = {'alpha': alpha_candidates}
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=lasso_base,
            param_grid=param_grid,
            cv=cv_folds,
            #the goal is to minimize RMSE, so we use negative RMSE for scoring
            scoring='neg_root_mean_squared_error',  # Use RMSE as scoring metric
            n_jobs=5,  # Use all available CPU cores for speed
            verbose=1   # Show progress
        )
        
        # Fit grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best parameters
        optimal_alpha = grid_search.best_params_['alpha']
        best_score = -grid_search.best_score_  # Convert back to positive RMSE
        
        print(f"Optimal alpha selected: {optimal_alpha:.6f}")
        print(f"Best cross-validation RMSE: {best_score:.4f}")
        
        # Print all scores for transparency
        print(f"\nAll cross-validation results:")
        for i, alpha_val in enumerate(alpha_candidates):
            mean_score = -grid_search.cv_results_['mean_test_score'][i]
            std_score = grid_search.cv_results_['std_test_score'][i]
            print(f"Alpha {alpha_val:8.6f}: RMSE = {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        # Use the best estimator (already fitted)
        lasso = grid_search.best_estimator_
        
    else:
        print(f"Using specified alpha: {alpha}")
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(X_train_scaled, y_train)
        optimal_alpha = alpha
    
    return lasso, scaler, optimal_alpha


def train_xgb_model(X_train, y_train, cv_folds=5, max_depth=None, n_estimators=None, 
                   max_depth_candidates=None, n_estimators_candidates=None):
    """
    Train XGBoost regression model with optional grid search cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        max_depth: Exact max_depth value (if specified, skip grid search for this parameter)
        n_estimators: Exact n_estimators value (if specified, skip grid search for this parameter)
        max_depth_candidates: List of max_depth values to test (if None, use default)
        n_estimators_candidates: List of n_estimators values to test (if None, use default)
        
    Returns:
        Fitted XGBoost model
        Best parameters dictionary
    """
    
    # Check if exact values are provided
    if max_depth is not None and n_estimators is not None:
        # Use exact values, no grid search needed
        print(f"Using specified XGBoost parameters: max_depth={max_depth}, n_estimators={n_estimators}")
        
        xgb_model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            objective='reg:squarederror',
            verbosity=0
        )
        
        # Fit the model directly
        xgb_model.fit(X_train, y_train)
        
        best_params = {
            'max_depth': max_depth,
            'n_estimators': n_estimators
        }
        
    else:
        # Use grid search for parameters not specified
        print(f"Performing {cv_folds}-fold grid search cross-validation for XGBoost...")
        
        # Define parameter candidates
        if max_depth is not None:
            max_depth_candidates = [max_depth]
        elif max_depth_candidates is None:
            max_depth_candidates = [3, 4, 5]
            
        if n_estimators is not None:
            n_estimators_candidates = [n_estimators]
        elif n_estimators_candidates is None:
            n_estimators_candidates = [5, 10, 20]
        
        print(f"Testing max_depth values: {max_depth_candidates}")
        print(f"Testing n_estimators values: {n_estimators_candidates}")
        
        # Create XGBoost model for grid search
        xgb_base = xgb.XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            verbosity=0  # Suppress XGBoost output
        )
        
        # Set up parameter grid
        param_grid = {
            'max_depth': max_depth_candidates,
            'n_estimators': n_estimators_candidates
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',  # Use RMSE as scoring metric
            n_jobs=5,  # Use multiple CPU cores for speed
            verbose=1   # Show progress
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)  # XGBoost doesn't need feature scaling
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back to positive RMSE
        
        print(f"Optimal parameters selected: {best_params}")
        print(f"Best cross-validation RMSE: {best_score:.4f}")
        
        # Print all scores for transparency
        print(f"\nAll cross-validation results:")
        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            params = results['params'][i]
            mean_score = -results['mean_test_score'][i]
            std_score = results['std_test_score'][i]
            print(f"max_depth={params['max_depth']}, n_estimators={params['n_estimators']}: RMSE = {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        # Use the best estimator (already fitted)
        xgb_model = grid_search.best_estimator_
    
    return xgb_model, best_params


def evaluate_model(model, scaler, X_test, y_test, model_name="model"):
    """
    Evaluate model performance for both LASSO and XGBoost models.
    
    Args:
        model: Trained model (LASSO or XGBoost)
        scaler: Fitted StandardScaler (used for LASSO, ignored for XGBoost)
        X_test: Test features
        y_test: Test target
        model_name: Name of the model type for proper handling
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Apply scaling only for LASSO models
    if model_name.lower() == "lasso":
        X_test_processed = scaler.transform(X_test)
    else:
        # XGBoost doesn't need feature scaling
        X_test_processed = X_test
    
    y_pred = model.predict(X_test_processed)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return metrics, y_pred


def export_feature_weights(model, feature_names, output_path, model_name="lasso"):
    """
    Export model feature weights/importance to CSV.
    
    Args:
        model: Trained model (LASSO or XGBoost)
        feature_names: List of feature names
        output_path: Path to save CSV file
        model_name: Type of model ('lasso' or 'xgboost')
    """
    
    if model_name.lower() == "lasso":
        # For LASSO: use coefficients
        weights = model.coef_
        abs_weights = np.abs(weights)
        is_selected = weights != 0
        weight_type = "coefficient"
    elif model_name.lower() == "xgboost":
        # For XGBoost: use feature importance
        # Get feature importance as a dict and convert to array in correct order
        importance_dict = model.get_booster().get_score(importance_type='weight')
        weights = np.array([importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_names))])
        abs_weights = weights  # Feature importance is already positive
        is_selected = weights > 0  # XGBoost feature importance is always >= 0
        weight_type = "importance"
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    weights_df = pd.DataFrame({
        'feature_name': feature_names,
        'weight': weights,
        'abs_weight': abs_weights,
        'is_selected': is_selected
    })
    
    # Sort by absolute weight (descending)
    weights_df = weights_df.sort_values('abs_weight', ascending=False)
    
    # Add ranking
    weights_df['weight_rank'] = range(1, len(weights_df) + 1)
    
    weights_df.to_csv(output_path, index=False)
    
    # Print summary
    selected_features = weights_df['is_selected'].sum()
    print(f"\nFeature {weight_type.title()} Summary ({model_name.upper()}):")
    print(f"Total features: {len(feature_names)}")
    
    if model_name.lower() == "lasso":
        print(f"Selected features (non-zero weights): {selected_features}")
        print(f"Sparsity: {(len(feature_names) - selected_features) / len(feature_names) * 100:.1f}%")
        display_df = weights_df[weights_df['is_selected']].head(10)
    else:
        print(f"Features with importance > 0: {selected_features}")
        display_df = weights_df.head(10)
    
    print(f"\nTop 10 features by {weight_type}:")
    print(display_df[['feature_name', 'weight', 'abs_weight']].to_string(index=False))
    
    return weights_df


def export_performance_metrics(metrics, model_params, data_info, output_path):
    """
    Export model performance metrics to CSV.
    
    Args:
        metrics: Dictionary with evaluation metrics (rmse, mae, r2, mse)
        model_params: Dictionary with model parameters (alpha, num_features_total, num_features_selected, etc.)
        data_info: Dictionary with data information (train_months, test_months, training_samples, test_samples)
        output_path: Path to save CSV file
    """
    
    # Combine all information into a single row
    metrics_data = {
        # Timestamp
        'experiment_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Data information
        'train_months': data_info.get('train_months', ''),
        'test_months': data_info.get('test_months', ''),
        'training_samples': data_info.get('training_samples', None),
        'test_samples': data_info.get('test_samples', None),

        # Performance metrics
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'mse': metrics['mse'],
        
        # Model parameters
        'num_features_total': model_params.get('num_features_total', None),
        'num_features_selected': model_params.get('num_features_selected', None),
        'sparsity_percent': ((model_params.get('num_features_total', 0) - model_params.get('num_features_selected', 0)) / 
                            model_params.get('num_features_total', 1) * 100) if model_params.get('num_features_total', 0) > 0 else 0,
    }
    # Ensure all model_params values are serializable before updating metrics_data
    metrics_data.update(model_params)  # Add all model params (e.g., alpha, max_depth, n_estimators)
    """
    serializable_params = {}
    for key, value in metrics_data.items():
        if isinstance(value, np.int64):
            serializable_params[key] = int(value)
        elif isinstance(value, np.float64):
            serializable_params[key] = float(value)
        else:
            serializable_params[key] = value
    metrics_data = serializable_params
    print(metrics_data)
    print(json.dumps(metrics_data, indent=2))
    """
    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics_data])
    
    # Save to CSV (append if file exists, otherwise create new)
    #if os.path.exists(output_path):
    #    # Read existing data and append
    #    existing_df = pd.read_csv(output_path)
    #    combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    #    combined_df.to_csv(output_path, index=False)
    #    print(f"Performance metrics appended to existing file: {output_path}")
    #else:
    # Create new file
    metrics_df.T.to_csv(output_path, index=True, header=False)
    print(f"Performance metrics saved to new file: {output_path}")
    
    print("\nPerformance Metrics Summary:")
    print(metrics_df.T.to_string(index=True, header=False))
    return metrics_df


def check_preprocessed_files_exist(outputs_dir):
    """Check if both preprocessed files exist in the outputs directory."""
    train_file = f"{outputs_dir}/preproc_full_train_df.parquet"
    test_file = f"{outputs_dir}/preproc_test_input_df.parquet"
    
    train_exists = os.path.exists(train_file)
    test_exists = os.path.exists(test_file)
    
    return train_exists, test_exists, train_file, test_file


def main():
    parser = argparse.ArgumentParser(description='LASSO Feature Selection for Building Optimization')
    
    # Configuration arguments
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                       help='Outputs directory (default: outputs)')
    parser.add_argument('--resample-freq-minutes', type=int, default=10,
                       help='Resampling frequency in minutes (default: 10)')
    parser.add_argument('--eps', type=float, default=1e-6,
                       help='Epsilon value (default: 1e-6)')
    parser.add_argument('--target-variable-name', type=str, default='B205WC000.AM02',
                       help='Target variable name (default: B205WC000.AM02)')
    
    # Data selection arguments
    parser.add_argument('--train-start-month', type=str, default='2025-01',
                       help='Training start month (format: YYYY-MM, default: 2025-01)')
    parser.add_argument('--train-end-month', type=str, default='2025-05',
                       help='Training end month (format: YYYY-MM, default: 2025-05)')
    parser.add_argument('--test-start-month', type=str, default='2025-05',
                       help='Test start month (format: YYYY-MM, default: 2025-05)')
    parser.add_argument('--test-end-month', type=str, default='2025-07',
                       help='Test end month (format: YYYY-MM, default: 2025-07)')
    
    # Feature selection arguments
    parser.add_argument('--all-features', action='store_true',
                       help='Use all available features except target variable')
    parser.add_argument('--predictor-variables', nargs='*', default=None,
                       help='Specific predictor variables to use (space-separated)')
    
    # Model arguments
    parser.add_argument('--models', nargs='*', type=str, default=['lasso'], 
                       choices=['lasso', 'xgboost'],
                       help='Models to train and evaluate (space-separated, choices: lasso, xgboost, default: lasso)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='LASSO regularization parameter (if not specified, use cross-validation)')
    parser.add_argument('--alpha-candidates', nargs='*', type=float, default=None,
                       help='Alpha candidates for grid search (space-separated floats, e.g., 10.0 1.0 0.1 0.01)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds for alpha selection (default: 5)')
    
    # XGBoost-specific arguments
    parser.add_argument('--max-depth', type=int, default=None,
                       help='XGBoost max depth parameter (if not specified, use grid search)')
    parser.add_argument('--n-estimators', type=int, default=None,
                       help='XGBoost n_estimators parameter (if not specified, use grid search)')
    parser.add_argument('--max-depth-candidates', nargs='*', type=int, default=None,
                       help='Max depth candidates for XGBoost grid search (space-separated ints, e.g., 3 4 5)')
    parser.add_argument('--n-estimators-candidates', nargs='*', type=int, default=None,
                       help='N estimators candidates for XGBoost grid search (space-separated ints, e.g., 5 10 20)')
    
    # Output arguments  
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data even if preprocessed files exist')
    
    args = parser.parse_args()
    
    # Note: Output paths are now model-specific and generated automatically:
    # - {outputs_dir}/{model_name}_feature_weights.csv
    # - {outputs_dir}/{model_name}_performance_metrics.csv
    
    # Create outputs directory
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    try:
        # Check for preprocessed files if not forcing reprocess
        if not args.force_reprocess:
            train_exists, test_exists, train_file, test_file = check_preprocessed_files_exist(args.outputs_dir)
            
            if train_exists and test_exists:
                print("Loading existing preprocessed files...")
                full_train_df = pd.read_parquet(train_file)
                test_input_df = pd.read_parquet(test_file)
                print("Preprocessed data loaded successfully.")
            else:
                print("Preprocessed files not found, processing raw data...")
                # Generate file paths based on month ranges
                train_file_paths = generate_file_paths_from_months(args.data_dir, args.train_start_month, args.train_end_month)
                test_file_paths = generate_file_paths_from_months(args.data_dir, args.test_start_month, args.test_end_month)
                
                # Process data
                full_train_df = simple_load_and_resample_data(
                    train_file_paths,
                    generate_sample_plots=[args.target_variable_name] + EXAMPLE_PREDICTOR_VARIABLE_NAMES,
                    save_load_df=train_file,
                    resample_freq_min=args.resample_freq_minutes,
                    outputs_dir=args.outputs_dir
                )
                test_input_df = simple_load_and_resample_data(
                    test_file_paths,
                    save_load_df=test_file,
                    resample_freq_min=args.resample_freq_minutes,
                    outputs_dir=args.outputs_dir
                )
        else:
            print("Force reprocessing enabled, processing raw data...")
            # Generate file paths based on month ranges
            train_file_paths = generate_file_paths_from_months(args.data_dir, args.train_start_month, args.train_end_month)
            test_file_paths = generate_file_paths_from_months(args.data_dir, args.test_start_month, args.test_end_month)
            
            # Process data
            full_train_df = simple_load_and_resample_data(
                train_file_paths,
                generate_sample_plots=[args.target_variable_name] + EXAMPLE_PREDICTOR_VARIABLE_NAMES,
                resample_freq_min=args.resample_freq_minutes,
                outputs_dir=args.outputs_dir
            )
            test_input_df = simple_load_and_resample_data(
                test_file_paths,
                resample_freq_min=args.resample_freq_minutes,
                outputs_dir=args.outputs_dir
            )
        
        print(f"Training data shape: {full_train_df.shape}")
        print(f"Test data shape: {test_input_df.shape}")
        
        # Prepare features and target
        predictor_vars = args.predictor_variables if args.predictor_variables else EXAMPLE_PREDICTOR_VARIABLE_NAMES
        X_train, y_train, feature_names, _ = prepare_features_and_target(
            full_train_df, 
            args.target_variable_name, 
            predictor_vars,
            args.all_features
        )
        
        X_test, y_test, _ = prepare_test_features_and_target(
            test_input_df, 
            args.target_variable_name, 
            feature_names  # Use training feature names to ensure consistency
        )
        
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Verify feature alignment
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: training has {X_train.shape[1]} features, test has {X_test.shape[1]} features")
        else:
            print(f"✓ Feature dimensions aligned: {X_train.shape[1]} features in both training and test sets")
        
        # Train and evaluate multiple models
        print(f"\nTraining and evaluating models: {args.models}")
        
        for model_name in args.models:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} model...")
            print(f"{'='*60}")
            
            if model_name.lower() == "lasso":
                # Train LASSO model
                lasso_model, scaler, optimal_alpha = train_lasso_model(
                    X_train, y_train, 
                    alpha=args.alpha, 
                    cv_folds=args.cv_folds,
                    alpha_candidates=args.alpha_candidates
                )
                
                # Evaluate LASSO model
                print(f"\nEvaluating {model_name.upper()} model...")
                metrics, y_pred = evaluate_model(lasso_model, scaler, X_test, y_test, model_name="lasso")
                
                # Model-specific parameters
                model_params = {
                    'alpha': optimal_alpha,
                    'num_features_total': len(feature_names),
                    'num_features_selected': (lasso_model.coef_ != 0).sum()
                }
                
                model = lasso_model
                
            elif model_name.lower() == "xgboost":
                # Train XGBoost model
                xgb_model, best_params = train_xgb_model(
                    X_train, y_train,
                    cv_folds=args.cv_folds,
                    max_depth=args.max_depth,
                    n_estimators=args.n_estimators,
                    max_depth_candidates=args.max_depth_candidates,
                    n_estimators_candidates=args.n_estimators_candidates
                )
                
                # Evaluate XGBoost model
                print(f"\nEvaluating {model_name.upper()} model...")
                metrics, y_pred = evaluate_model(xgb_model, None, X_test, y_test, model_name="xgboost")
                
                # Model-specific parameters
                model_params = {
                    'max_depth': best_params['max_depth'],
                    'n_estimators': best_params['n_estimators'],
                    'num_features_total': len(feature_names),
                    'num_features_selected': (xgb_model.feature_importances_ > 0).sum()
                }
                
                model = xgb_model
                scaler = None  # XGBoost doesn't use scaling
            
            # Generate model-specific output paths
            weights_output = f"{args.outputs_dir}/{model_name}_feature_weights.csv"
            metrics_output = f"{args.outputs_dir}/{model_name}_performance_metrics.csv"
            
            # Export feature weights/importance
            print(f"\nExporting feature weights/importance to {weights_output}...")
            weights_df = export_feature_weights(model, feature_names, weights_output, model_name=model_name)
            
            # Export performance metrics
            print(f"Exporting performance metrics to {metrics_output}...")
            data_info = {
                'train_months': f"{args.train_start_month} to {args.train_end_month}",
                'test_months': f"{args.test_start_month} to {args.test_end_month}",
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            metrics_df = export_performance_metrics(metrics, model_params, data_info, metrics_output)
        
        print("\nDone.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())