"""This is a script that can serve as an initialization for participating in the
ELIAS Bosch AI for Building Optimisation prediction competition.

It contains simple versions of the essential components for
* loading and preprocessing data,
* defining a simple torch pairs dataset for causal prediction (i.e., using only past to predict future),
* defining a simple toy example model and training it,
* and evaluating and creating the submission file (`submission_file.csv`) of that model on the test input data.

Note that all of the components are just starting points, and many aspects can still be improved, see also `README.md`.

This is a custom version of main.py with optimized preprocessing that skips raw file processing
if preprocessed files already exist.
"""

# Copyright (c) 2025 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
from datetime import datetime, date
import glob
from itertools import chain
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import wandb
import sys

from feature_groups import add_room_return_temp_features

def setup_device(device_arg):
    """Setup device based on argument.
    
    Args:
        device_arg: Device specification (None, 'cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        str: The device string to use
    """
    if device_arg is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
        # Validate device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    
    return device

def setup_wandb():
    """Setup wandb authentication and return API key status.
    
    Returns:
        bool: True if wandb is successfully set up, False otherwise
    """
    try:
        # Try to read API key from file
        api_key_file = "wandb_api_key.txt"
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            wandb.login(key=api_key)
            print("Successfully logged into wandb using API key from file")
            return True
        else:
            print(f"Warning: {api_key_file} not found. Proceeding without wandb logging.")
            return False
    except Exception as e:
        print(f"Warning: Failed to setup wandb: {e}. Proceeding without wandb logging.")
        return False


DATA_DIR = "data"
OUTPUTS_DIR = os.path.join(DATA_DIR, "splits")
RESAMPLE_FREQ_MIN = 10  # the frequency in minutes to resample the raw irregularly sampled timeseries to, using ffill
EPS = 1e-6
TARGET_VARIABLE_NAME = "B205WC000.AM02"  # the target variable to be predicted
EXAMPLE_PREDICTOR_VARIABLE_NAMES = [
    #ORIGINAL 2:
    "B205WC000.AM01",  # a supply temperature chilled water
    "B106WS01.AM54",  # an external temperature - most highly correlated weather feature
    #high abs corr weather: in globally low corr features
    #'B106WS01.AM51',  # light intensity
    #'B106WS01.AM53',  # humidity
    #high abs corr new analysis: above 0.5 abs corr with shift -180min
    #'B205WC001.AM71',# TOTAL VOLUME CHILLED WATER
    #'B205WC000.AM71',# VOLUME CHILLED WATER BP201/202/206
    #'B205WC140.AC21',# PRIMARY VALVE 1
    #'B201RC572.AC61',# VAV SUPPLY AIR 201.C.571
    #high abs corr new analysis: above 0.4 abs corr with shift -180min
    #'B201FC149_1.VT03_2',# ACTIVE SETPOINT TEMP. 201.A.034
    #'B201FC149_1.VT03_1',# USER SETPOINT TEMP. 201.A.034
    ##'B205WC003.AM02',# RETURN TEMP, CHILLED WATER BP201/202/206 - this is not in the data
    #'B201FC149_1.VS01_1',# STATUS STEP VENTILATION 201.A.034
    #'B205WC002.RA001',# SPEED CHILLED WATER PUMP
    #'B201AH163.AC21',# COOLER VALVE
    #'B201FC223_1.VT03_1',# USER SETPOINT TEMP. 201.A.287
    #'B201RC055.AM01',# ROOM TEMPERATURE 201.B.074b
    #'B201FC096_1.AM01',# ROOM TEMPERATURE 201.C.035
    ##'B205WC001.AM02',# RETURN TEMPERATURE CHILLED WATER - this is not in the data
    #'B201AH601.AM15',# RETURN-AIR TEMPERATURE
    #'B201AH164.AC21',# COOLER VALVE
    #'B201RC044.AM02',# ROOM TEMPERATURE 2 201.B.080 Z.6
    #'B201FC288_1.VT03_1',# USER SETPOINT TEMP. 201.C.287
    #'B201FC403_4.AM01',# ROOM TEMPERATURE 201.A.432
    #'B201FC664_2.VT02_1',# ACTIVE SP. TEMP. COOLING 201.C.634
    ##'B205WC140.AM04',# INLET TEMPERATURE HEAT EXCHANGER SEC. - this is not in the data
]

#EXAMPLE_PREDICTOR_VARIABLE_NAMES += external_measurements

EXAMPLE_PREDICTOR_VARIABLE_NAMES = list(set(EXAMPLE_PREDICTOR_VARIABLE_NAMES))# remove duplicates
SUBMISSION_FILE_PATH = f"{OUTPUTS_DIR}/submission_file.csv"
SUBMISSION_FILE_TARGET_VARIABLE_COLUMN_NAME = "TARGET_VARIABLE"
SUBMISSION_FILE_DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def simple_load_and_resample_data(
    data_file_paths, generate_sample_plots=None, save_load_df=None
):
    """Load measurement timeseries data from the original data folder structure, turn the original irregularly sampling
    into a regular, common timeline, and join everything into one multivariate timeseries dataframe.

    Important: might have to be adapted for actual models for competition.

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
        for path in tqdm(data_file_paths, mininterval=15):
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
            resample_freq_str = f"{str(RESAMPLE_FREQ_MIN)}min"
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
            figsize=(10, 4 * n_plots)
        )
        plt.title("Input data timeseries")
        for i, col in enumerate(generate_sample_plots):
            axs[i].plot(multivariate_timeseries_df[col], label=col, linewidth=0.75)
            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].legend(fontsize="small")
        # Extract output directory from save_load_df path
        if save_load_df:
            output_dir = os.path.dirname(save_load_df)
        else:
            output_dir = OUTPUTS_DIR
        plt.savefig(os.path.join(output_dir, "input_data_sample_timeseries_plot.png"), bbox_inches="tight")
        plt.close(fig)

    if save_load_df and save:
        multivariate_timeseries_df.to_parquet(save_load_df)

    print("Done.")

    return multivariate_timeseries_df


def simple_eval_and_submission_creation(
    loader,
    model,
    loss_fn=None,
    generate_timeseries_prediction=False,
    save_fig=None,
    create_submission_df=False,
):
    """Evaluate the model on the given data loader and optionally create a coherent prediction on the full dataset
    underlying the dataloader, and save it as a figure.

    Important: this might have to be adapted for actual models for competition.

    Args:
        loader: The data loader to run and evaluate the model on. It gives (x, y) pairs, i.e., input variable and target
            variable, where the to-be-predicted target variable value is allowed to be nan, for the sake of just
            producing a prediction for submission (without having the ground truth target variable value available right
            here).
        model: The model to evaluate.
        loss_fn: The loss function to use for evaluation. If None, then this function runs in the mode of just producing
            the prediction for submission.
        generate_timeseries_prediction: Whether to generate a timeseries prediction from the model.
            If True, the predictions will be concatenated and returned as a single tensor.
        save_fig: If provided, the path to save the figure of the predictions.
            If None, no figure will be saved.
        create_submission_df: If True, a DataFrame will be created from the predictions,
            with the index being the datetime of the timestamp in the first column of the tensor `y_pred`. Use this
            option to afterwards create a submission file that can then be submitted to the competition.
            If a datetime is provided, the DataFrame will only contain entries from that datetime onwards.

    Returns:
        A dictionary containing the evaluation results. If `generate_timeseries_prediction` is True, it will contain
        the predictions as a tensor under the key "ys_pred". If `create_submission_df` is True, it will also contain
        the predictions as a DataFrame under the key "ys_pred_df". If `loss_fn` is provided, it will also
        contain the average loss under the key "avg_loss".
    """
    res = {}

    model.eval()
    ys_true_list = []
    ys_pred_list = []
    if loss_fn is not None:
        cum_batch_loss_list = []
    with torch.no_grad():
        for x_true, y_true in loader:
            assert ~x_true.isnan().any()
            y_pred = model(x_true)
            assert (y_pred[:, 0] == y_true[:, 0]).all(), (
                "Annotated timestamp of prediction does not match to-be-predicted timestamp."
            )
            y_true_core, y_pred_core = (
                y_true[:, 1:],
                y_pred[:, 1:],
            )  # Exclude timestamp from loss calculation
            ys_true_list.append(y_true)
            ys_pred_list.append(y_pred)
            if loss_fn is not None:
                assert loss_fn.reduction == "mean", (
                    "Loss function must have reduction='mean', because the calculation assumes so."
                )
                batch_size = y_true.shape[0]
                assert not torch.isnan(y_true).any(), (
                    "y_true contains NaN values, which is not allowed for loss computation. "
                    "Are you sure the loader is for train/vali, and not submission, "
                    "where the y is just an empty dummy entry?"
                )
                loss = loss_fn(y_pred_core, y_true_core)
                cum_loss = (
                    loss * batch_size
                )  # multiply by batch size to be invariant to batch size
                cum_batch_loss_list.append(cum_loss.unsqueeze(0))
        if loss_fn is not None:
            assert loader.drop_last is False, (
                "Loader must not drop last incomplete batch, otherwise the loss calculation does not work properly."
            )
            avg_loss = torch.cat(cum_batch_loss_list).sum() / len(loader.dataset)
            res["avg_loss"] = avg_loss

    if generate_timeseries_prediction:
        assert isinstance(loader.sampler, torch.utils.data.sampler.SequentialSampler), (
            "Loader must not shuffle data, otherwise the concatenated batch predictions "
            "do not form a properly ordered time series prediction."
        )
        ys_true = torch.cat(ys_true_list, dim=0)
        ys_pred = torch.cat(ys_pred_list, dim=0)
        res["ys_pred"] = ys_pred

        if save_fig is not None or create_submission_df:
            ys_true_df = pd.DataFrame(
                ys_true[:, 1].cpu().numpy(),
                index=pd.to_datetime(ys_true[:, 0].cpu().numpy(), unit="s"),
                columns=["Value(true)"],
            )
            ys_pred_df = pd.DataFrame(
                ys_pred[:, 1].cpu().numpy(),
                index=pd.to_datetime(ys_pred[:, 0].cpu().numpy(), unit="s"),
                columns=[SUBMISSION_FILE_TARGET_VARIABLE_COLUMN_NAME],
            )
            ys_pred_df.index.name = "time"

            if save_fig is not None:
                fig, ax = plt.subplots(dpi=500)
                ax.plot(ys_true_df, label="true", alpha=0.75, linewidth=0.75)
                ax.plot(ys_pred_df, label="pred", alpha=0.75, linewidth=0.75)
                ax.tick_params(axis="x", labelrotation=90)
                ax.legend(fontsize="small")
                plt.savefig(save_fig, bbox_inches="tight")
                plt.close(fig)
            if create_submission_df:
                if create_submission_df == True:
                    res["ys_pred_df"] = ys_pred_df
                elif isinstance(create_submission_df, datetime):
                    res["ys_pred_df"] = ys_pred_df[
                        ys_pred_df.index >= create_submission_df
                    ]
    return res

def simple_feature_dataset(
    full_multivariate_timeseries_df, add_dummy_y=False, normalize=False, feature_hours=1, input_seq_step=1, stride=1, use_custom_date_features=False, use_custom_sensor_features=False
):
    """
    Create a simple feature dataset from the full multivariate timeseries dataframe.
    Parameters:
        full_multivariate_timeseries_df: The full multivariate timeseries dataframe.
        add_dummy_y: Whether to add a dummy target variable column with NaN values.
        normalize: Whether to normalize the features. If True, the mean and std will be computed from the data.
            If a dict is provided, it should contain 'timeseries_df_mean' and 'timeseries_df_std' for normalization.
        feature_hours: The number of past hours to use as features.
        input_seq_step: The step size for the input sequence. You might not use every time step from the last `feature_hours`.
        stride: The stride for moving the input window. Might not want to train on every available time step.
    """
    info = {}

    input_seq_len = int( int(60 / RESAMPLE_FREQ_MIN) * feature_hours)  # hours
    predict_ahead = int(60 / RESAMPLE_FREQ_MIN) * 3  # hours

    # restrict to only relevant/valid data:
    timeseries_df = full_multivariate_timeseries_df[
        [
            col
            for col in EXAMPLE_PREDICTOR_VARIABLE_NAMES + [TARGET_VARIABLE_NAME]
            if col in full_multivariate_timeseries_df.columns
        ]
    ].copy()
    
    #WHY is it needed? probably due to ffill at beginning? (for now let's keep it commented - as we try to include more features)
    #first_valid_idx = timeseries_df.notna().all(axis=1).idxmax()
    #timeseries_df = timeseries_df.loc[first_valid_idx:]
    timeseries_df.ffill(inplace=True)
    if isinstance(normalize, dict):
        # for the test set, use mean from training set:
        timeseries_df.fillna(normalize["timeseries_df_mean"], inplace=True)
    else:
        timeseries_df.fillna(timeseries_df.mean(), inplace=True)

    if add_dummy_y:
        timeseries_df[TARGET_VARIABLE_NAME] = np.nan

    # extract and add some features:
    datetime_features = []
    datetime_list = timeseries_df.index.to_pydatetime()
    timeseries_df["timestamp"] = [dtime.timestamp() for dtime in datetime_list]
    #timestamp is not added to datetime_features, it is handled separately later
    
    timeseries_df["minute_of_day"] = (
        timeseries_df.index.hour * 60 + timeseries_df.index.minute
    )
    datetime_features.append("minute_of_day")

    timeseries_df["day_of_week"] = timeseries_df.index.dayofweek
    datetime_features.append("day_of_week")
    
    timeseries_df["day_of_year"] = timeseries_df.index.dayofyear#FIX: originally it was also dayofweek call in this line
    datetime_features.append("day_of_year")
    
    timeseries_df["yeartime_sin"] = np.sin(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    datetime_features.append("yeartime_sin")
    timeseries_df["yeartime_cos"] = np.cos(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    datetime_features.append("yeartime_cos")

    timeseries_df["daytime_sin"] = np.sin(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )
    datetime_features.append("daytime_sin")
    timeseries_df["daytime_cos"] = np.cos(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )
    datetime_features.append("daytime_cos")

    # NEW DATETIME FEATURES
    if use_custom_date_features:
        timeseries_df["weektime_sin"] = np.sin(
            2 * np.pi * timeseries_df["day_of_week"] / 7
        )
        datetime_features.append("weektime_sin")
        timeseries_df["weektime_cos"] = np.cos(
            2 * np.pi * timeseries_df["day_of_week"] / 7
        )
        datetime_features.append("weektime_cos")
    
        from utils import HungarianWorkdayAnalyzer
        # Initialize global analyzer instance
        hungarian_analyzer = HungarianWorkdayAnalyzer()

        # Hungarian holiday and working day features
        timeseries_df["is_hungarian_holiday"] = timeseries_df.index.to_series().apply(hungarian_analyzer.is_official_holiday).astype(int)
        timeseries_df["is_working_day"] = timeseries_df.index.to_series().apply(hungarian_analyzer.is_working_day).astype(int)
        timeseries_df["is_weekend"] = timeseries_df.index.to_series().apply(hungarian_analyzer.is_weekend).astype(int)
        datetime_features.extend(["is_hungarian_holiday", "is_working_day", "is_weekend"])
        #print(timeseries_df['is_hungarian_holiday'].value_counts())
        #print(timeseries_df['is_working_day'].value_counts())
        #print(timeseries_df['is_weekend'].value_counts())

    non_date_features = EXAMPLE_PREDICTOR_VARIABLE_NAMES.copy()
    if use_custom_sensor_features:
        metadata = pd.read_parquet(os.path.join(DATA_DIR, 'kaggle_dl', 'metadata.parquet'))
        timeseries_df, new_feature_cols = add_room_return_temp_features(metadata, timeseries_df)
        non_date_features += new_feature_cols

    column_names = timeseries_df.columns
    print('Dataset columns:', column_names.tolist())
    
    # Store feature information in info dict
    info['feature_columns'] = column_names.tolist()
    info['predictor_variables'] = EXAMPLE_PREDICTOR_VARIABLE_NAMES
    info['target_variable'] = TARGET_VARIABLE_NAME
    info['datetime_features'] = datetime_features
    info['feature_hours'] = feature_hours
    info['input_seq_step'] = input_seq_step
    info['stride'] = stride
    info['use_custom_date_features'] = use_custom_date_features
    info['input_seq_len'] = input_seq_len
    info['predict_ahead'] = predict_ahead

    if normalize:
        if normalize == True:
            mean = timeseries_df.mean()
            std = timeseries_df.std()
            info.update(
                {
                    "timeseries_df_mean": mean,
                    "timeseries_df_std": std,
                }
            )
        elif isinstance(normalize, dict):
            mean = normalize["timeseries_df_mean"]
            std = normalize["timeseries_df_std"]

        def normalization_fn(x, col):
            return (x - torch.tensor(mean[col])) / (torch.tensor(std[col]) + EPS)
    else:

        def normalization_fn(x, col):
            return x

    timeseries_tensor = torch.tensor(
        timeseries_df.to_numpy(), dtype=torch.get_default_dtype()
    )
    data = timeseries_tensor

    X = []
    Y = []
    for i in range(0, data.shape[0] - input_seq_len - predict_ahead, stride):
        selected_features = []
        
        #first come timestamp
        timestamp = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("timestamp")
        ].unsqueeze(0)
        selected_features.append(timestamp)
        # next: datetime features:
        for dt_feature in datetime_features:
            if dt_feature == "timestamp":
                continue  # already handled
            elif dt_feature == "day_of_week":
                day_of_week = torch.nn.functional.one_hot(
                    data[
                        i + input_seq_len + predict_ahead, column_names.get_loc("day_of_week")
                    ].to(dtype=torch.long),
                    num_classes=7,
                )
                selected_features.append(day_of_week)
            else:
                dt_feature_values = data[
                    i + input_seq_len + predict_ahead, column_names.get_loc(dt_feature)
                ].unsqueeze(0)
                selected_features.append(dt_feature_values)

        # finally: not date related predictor variables:
        for predictor in non_date_features:
            if not predictor in column_names:
                raise ValueError(
                    f"Predictor variable {predictor} not found in data columns."
                )
            predictor_values = normalization_fn(
                data[
                    i : i + input_seq_len : input_seq_step,
                    column_names.get_loc(predictor),
                ],
                predictor,
            )
            selected_features.append(predictor_values)

        # Concatenate all selected features
        X.append(torch.cat(selected_features))

        target_variable = data[
            i + input_seq_len + predict_ahead,
            column_names.get_loc(TARGET_VARIABLE_NAME),
        ].unsqueeze(0)

        Y.append(torch.cat([timestamp, target_variable]))

    X = torch.stack(X)
    Y = torch.stack(Y)
    print('Tensor shapes:', X.shape, Y.shape)
    dataset = TensorDataset(X, Y)

    return dataset, info


def simple_model_and_train(train_loader, vali_loader, loss_fn, use_wandb=False):
    """Define a simple prediction model and train it on the given training data loader.

    Important: to be adapted for actual models for competition.
    
    Args:
        train_loader: Training data loader
        vali_loader: Validation data loader  
        loss_fn: Loss function to use
        use_wandb: Whether to log metrics to wandb
    """

    class SimpleAIFBOModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.mlp = torchvision.ops.MLP(
                in_channels=input_size,
                hidden_channels=[128, 128, 1],
                norm_layer=nn.LayerNorm,
            ).to(dtype=torch.get_default_dtype())

        def forward(self, x):
            timestamp_of_prediction, x_core = x[:, :1], x[:, 1:]
            y_core = self.mlp(x_core)
            return torch.cat([timestamp_of_prediction, y_core], dim=1)

    x, _ = next(iter(train_loader))
    input_size = (
        x.shape[-1] - 1
    )  # Get the input size from the first batch, subtract 1 for the timestamp

    model = SimpleAIFBOModel(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    for epoch in range(200):
        model.train()
        cum_batch_loss_list = []
        for _, (x_true, y_true) in enumerate(train_loader):
            assert ~x_true.isnan().any() and ~y_true.isnan().any()
            y_pred = model(x_true)
            assert (y_pred[:, 0] == y_true[:, 0]).all(), (
                "Annotated timestamp of prediction does not match to-be-predicted timestamp."
            )
            y_true_core, y_pred_core = (
                y_true[:, 1:],
                y_pred[:, 1:],
            )  # Exclude timestamp from loss calculation
            loss = loss_fn(y_pred_core, y_true_core)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = y_true.shape[0]
            cum_loss = (
                loss * batch_size
            )  # multiply by batch size to be invariant to batch size
            cum_batch_loss_list.append(cum_loss.unsqueeze(0))
        avg_train_loss_running = torch.cat(cum_batch_loss_list).sum() / len(
            train_loader.dataset
        )

        avg_train_loss_epoch = simple_eval_and_submission_creation(
            train_loader, model, loss_fn
        )["avg_loss"]
        avg_vali_loss = simple_eval_and_submission_creation(
            vali_loader, model, loss_fn
        )["avg_loss"]

        print(
            f"Epoch: {epoch:04d}."
            f"Train Loss: {avg_train_loss_running:.5f}. "
            f"Train Loss Epoch: {avg_train_loss_epoch:.5f}. "
            f"Vali Loss: {avg_vali_loss:.5f}"
        )
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss_running": avg_train_loss_running.item(),
                "train_loss_epoch": avg_train_loss_epoch.item(),
                "validation_loss": avg_vali_loss.item(),
            })

    return model

def sklearn_model_and_train(models, train_loader, vali_loader, cv=5, use_wandb=False):
    """Define and train sklearn models on the given training data loader using cross-validation.

    Important: to be adapted for actual models for competition.
    
    Args:
        models: Dictionary of sklearn models to train
        train_loader: Training data loader
        vali_loader: Validation data loader  
        cv: Number of cross-validation folds
        use_wandb: Whether to log metrics to wandb
    """

    #from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import TimeSeriesSplit

    x_train_list = []
    y_train_list = []
    for x, y in train_loader:
        x_train_list.append(x[:, 1:].cpu().numpy())  # Exclude timestamp
        y_train_list.append(y[:, 1].cpu().numpy())  # Exclude timestamp

    X_train = np.vstack(x_train_list)
    y_train = np.hstack(y_train_list)

    model_performance = {}
    best_model_name = None
    best_score = float('inf')
    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        tscv = TimeSeriesSplit(n_splits=cv)
        val_scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            model.fit(X_train_fold, y_train_fold)
            val_score = np.mean((model.predict(X_val_fold) - y_val_fold) ** 2)
            val_scores.append(val_score)
        avg_val_score = np.mean(val_scores)
        model_performance[model_name] = avg_val_score
        print(f"Average Validation Score for {model_name}: {avg_val_score:.5f}")

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                f"{model_name}_validation_score": avg_val_score,
            })

        if avg_val_score < best_score:
            best_score = avg_val_score
            best_model_name = model_name
    return model_performance

def get_data_file_paths(train_start, train_end, test_start=None, test_end=None):
    """Generate file paths based on date ranges. Note that for the test set, one month before the test_start is also included for proper feature extraction.
    
    Args:
        train_start: Training start date in YYYY-MM format
        train_end: Training end date in YYYY-MM format
        test_start: Test start date in YYYY-MM format. If None, defaults to the month after train_end.
        test_end: Test end date in YYYY-MM format. If None, defaults to the month after test_start.

    Returns:
        tuple: (train_file_paths, test_file_paths, test_start_datetime)
    """
    def get_next_month(date_str):
        year, month = map(int, date_str.split('-'))
        if month == 12:
            return f"{year + 1}-01"
        else:
            return f"{year}-{month + 1:02d}"
        
    def get_previous_month(date_str):
        year, month = map(int, date_str.split('-'))
        if month == 1:
            return f"{year - 1}-12"
        else:
            return f"{year}-{month - 1:02d}"

    def get_month_paths(start_date, end_date):
        """Get file paths for months in range."""
        paths = []
        start_year, start_month = map(int, start_date.split('-'))
        end_year, end_month = map(int, end_date.split('-'))
        
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            month_pattern = f"{DATA_DIR}/kaggle_dl/RBHU-{current_year:04d}-{current_month:02d}/RBHU/**/*.parquet"
            paths.extend(glob.glob(month_pattern, recursive=True))
            
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        return paths
    
    train_file_paths = get_month_paths(train_start, train_end)

    if test_start is None or test_end is None:
        print("No test date range provided, by default using next two months after training end date.")
        #test_start = get_next_month(train_end)
        #test_end = get_next_month(test_start)
        test_start_datetime = None
        test_file_paths = []
    else:
        # Create test start datetime from test_start parameter
        test_year, test_month = map(int, test_start.split('-'))
        test_start_datetime = datetime(test_year, test_month, 1)

        # we load one month before test start for proper feature extraction:
        test_file_paths = get_month_paths(get_previous_month(test_start), test_end)
    
    return train_file_paths, test_file_paths, test_start_datetime


def get_data_split_folder(train_start, train_end, test_start, test_end):
    """Generate folder name for data split."""
    return f"train_{train_start}_to_{train_end}_test_{test_start}_to_{test_end}"


def check_preprocessed_files_exist(data_split_folder):
    """Check if both preprocessed files exist in the data split directory.
    
    Args:
        data_split_folder: Name of the data split subfolder
    
    Returns:
        bool: True if both files exist, False otherwise
    """
    data_split_dir = os.path.join(OUTPUTS_DIR, data_split_folder)
    train_file = os.path.join(data_split_dir, "preproc_full_train_df.parquet")
    test_file = os.path.join(data_split_dir, "preproc_test_input_df.parquet")
    
    train_exists = os.path.exists(train_file)
    test_exists = os.path.exists(test_file)
    
    if train_exists and test_exists:
        print("Both preprocessed files found:")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        return True
    else:
        if not train_exists:
            print(f"Preprocessed train file not found: {train_file}")
        if not test_exists:
            print(f"Preprocessed test file not found: {test_file}")
        return False

def run_resample_mode(train_start, train_end, test_start=None, test_end=None):
    """Run data resampling mode: load raw data, resample, and save to parquet files."""
    print("Running in RESAMPLE mode...")
    
    # Get file paths based on date ranges
    train_file_paths, test_file_paths, test_start_datetime = get_data_file_paths(
        train_start, train_end, test_start, test_end
    )
    
    print(f"Found {len(train_file_paths)} training files")
    print(f"Found {len(test_file_paths)} test files") 
    
    if len(train_file_paths) == 0:
        print("Error: No training files found for the specified date range!")
        sys.exit(1)
    
    if len(test_file_paths) == 0:
        print("Error: No test files found for the specified date range!")
        #sys.exit(1)
    
    # Create data split folder
    data_split_folder = get_data_split_folder(train_start, train_end, test_start, test_end)
    data_split_dir = os.path.join(OUTPUTS_DIR, data_split_folder)
    os.makedirs(data_split_dir, exist_ok=True)
    
    print(f"Data split folder: {data_split_dir}")
    
    # Process training data
    print("Processing training data...")
    train_save_path = os.path.join(data_split_dir, "preproc_full_train_df.parquet")
    full_train_df = simple_load_and_resample_data(
        train_file_paths,
        generate_sample_plots=[TARGET_VARIABLE_NAME] + EXAMPLE_PREDICTOR_VARIABLE_NAMES,
        save_load_df=train_save_path,
    )
    
    if len(test_file_paths) > 0:
        # Process test data  
        print("Processing test data...")
        test_save_path = os.path.join(data_split_dir, "preproc_test_input_df.parquet")
        test_input_df = simple_load_and_resample_data(
            test_file_paths,
            save_load_df=test_save_path,
        )
    
    # Save metadata about the data split
    metadata = {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,  
        'test_end': test_end,
        'test_start_datetime': test_start_datetime.isoformat() if test_start_datetime else None,
        'train_files_count': len(train_file_paths),
        'test_files_count': len(test_file_paths),
        'train_shape': list(full_train_df.shape),
        'test_shape': list(test_input_df.shape) if len(test_file_paths) > 0 else None,
        'train_time_min': full_train_df.index.min().isoformat(),
        'train_time_max': full_train_df.index.max().isoformat(),
        'test_time_min': test_input_df.index.min().isoformat() if len(test_file_paths) > 0 else None,
        'test_time_max': test_input_df.index.max().isoformat() if len(test_file_paths) > 0 else None,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(data_split_dir, "split_metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data resampling completed successfully!")
    print(f"Training data shape: {full_train_df.shape}")
    if len(test_file_paths) > 0:
        print(f"Test data shape: {test_input_df.shape}")
    print(f"Files saved to: {data_split_dir}")
    print(f"Metadata saved to: {metadata_path}")


def run_experiment_mode(data_split, args):
    """Run experiment mode: load preprocessed data and run ML experiment."""
    print("Running in EXPERIMENT mode...")
    
    data_split_dir = os.path.join(OUTPUTS_DIR, data_split)
    
    # Check if preprocessed files exist
    #if not check_preprocessed_files_exist(data_split):
    #    print(f"Error: Preprocessed files not found in {data_split_dir}")
    #    print("Please run resample mode first to generate the data.")
    #    sys.exit(1)
    
    # Load metadata
    metadata_path = os.path.join(data_split_dir, "split_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        test_start_datetime = metadata['test_start_datetime']
        if test_start_datetime:
            test_start_datetime = datetime.fromisoformat(test_start_datetime)

        print(f"Loaded data split metadata: {metadata['train_start']} to {metadata['train_end']} (train), {metadata['test_start']} to {metadata['test_end']} (test)")
    else:
        raise RuntimeError("metadata.json not found, using default test start datetime")
    
    # Load preprocessed data
    print("Loading preprocessed train data...")
    train_file = os.path.join(data_split_dir, "preproc_full_train_df.parquet")
    full_train_df = pd.read_parquet(train_file)
    
    test_file = os.path.join(data_split_dir, "preproc_test_input_df.parquet")
    if os.path.exists(test_file):
        print("Loading preprocessed test data...")
        test_input_df = pd.read_parquet(test_file)
    else:
        test_input_df = None
    
    print("Preprocessed data loaded successfully.")
    print(f"Training data shape: {full_train_df.shape}")
    if test_input_df is not None:
        print(f"Test data shape: {test_input_df.shape}")
    
    # Continue with the rest of the experiment pipeline
    tzinfo = full_train_df.index.tzinfo

    # Turn it into torch datasets for simple prediction from past to future, with simple features:
    full_train_dataset, full_train_dataset_info = simple_feature_dataset(
        full_train_df, add_dummy_y=False, normalize=True, feature_hours=args.feature_hours, 
        input_seq_step=args.input_seq_step, stride=args.stride, use_custom_date_features=args.use_custom_date, use_custom_sensor_features=args.use_custom_sensor
    )

    if test_input_df is not None:
        #stride is set to 1 for test dataset, because we want predictions for every time step in test set
        test_input_dataset, _ = simple_feature_dataset(
            test_input_df, add_dummy_y=True, normalize=full_train_dataset_info, feature_hours=args.feature_hours, 
            input_seq_step=args.input_seq_step, stride=1, use_custom_date_features=args.use_custom_date, use_custom_sensor_features=args.use_custom_sensor
        )
        test_input_loader = DataLoader(test_input_dataset, batch_size=64, shuffle=False)

    # Turn it into data loaders for training, validation, and submission (where submission loader differs in that
    # it has no target variable values, i.e. y):
    len_full_train_dataset = len(full_train_dataset)
    split_index = int(0.8 * len_full_train_dataset)
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(split_index))
    vali_dataset = torch.utils.data.Subset(
        full_train_dataset, range(split_index, len_full_train_dataset)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()),
    )
    vali_loader = DataLoader(vali_dataset, batch_size=64, shuffle=False)

    # Setup output directory for this experiment
    experiment_output_dir = os.path.join(data_split_dir, "experiments", args.name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Export experiment parameters and features
    experiment_params = {
        'experiment_name': args.name,
        'data_split': data_split,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'sklearn' if args.sklearn else 'pytorch',
        'feature_parameters': {
            'feature_hours': getattr(args, 'feature_hours', 1),
            'input_seq_step': getattr(args, 'input_seq_step', 1),
            'stride': getattr(args, 'stride', 1),
            'use_custom_date_features': getattr(args, 'use_custom_date', False),
            'resample_freq_min': RESAMPLE_FREQ_MIN,
            'input_seq_len': full_train_dataset_info.get('input_seq_len'),
            'predict_ahead': full_train_dataset_info.get('predict_ahead'),
        },
        'data_info': {
            'train_shape': list(full_train_df.shape),
            'test_shape': list(test_input_df.shape) if test_input_df is not None else None,
            'target_variable': TARGET_VARIABLE_NAME,
            'predictor_variables': EXAMPLE_PREDICTOR_VARIABLE_NAMES,
            'datetime_features': full_train_dataset_info.get('datetime_features', []),
        },
        'dataset_info': {
            'train_dataset_length': len(full_train_dataset),
            'test_dataset_length': len(test_input_dataset),
            'train_split_index': int(0.8 * len(full_train_dataset)),
        }
    }
    
    # Save experiment parameters as JSON
    params_file = os.path.join(experiment_output_dir, "experiment_parameters.json")
    import json
    with open(params_file, 'w') as f:
        json.dump(experiment_params, f, indent=2)
    
    # Save feature lists as text files
    train_features_file = os.path.join(experiment_output_dir, "train_features.txt")
    with open(train_features_file, 'w') as f:
        f.write("Training Data Features:\n")
        f.write("=" * 50 + "\n")
        for i, feature in enumerate(full_train_dataset_info.get('feature_columns', []), 1):
            f.write(f"{i:3d}. {feature}\n")
        f.write(f"\nTotal features: {len(full_train_dataset_info.get('feature_columns', []))}\n")
    
    test_features_file = os.path.join(experiment_output_dir, "test_features.txt")  
    with open(test_features_file, 'w') as f:
        f.write("Test Data Features:\n")
        f.write("=" * 50 + "\n")
        # Test features should be the same as train features
        for i, feature in enumerate(full_train_dataset_info.get('feature_columns', []), 1):
            f.write(f"{i:3d}. {feature}\n")
        f.write(f"\nTotal features: {len(full_train_dataset_info.get('feature_columns', []))}\n")
    
    print(f"📄 Experiment parameters saved:")
    print(f"  Parameters: {params_file}")
    print(f"  Train features: {train_features_file}")
    print(f"  Test features: {test_features_file}")
    
    if args.sklearn:
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from xgboost import XGBRegressor

        # Define and train sklearn models
        models = {
            #"LinearRegression": LinearRegression(),without feature selection it is sensitive to outliers
            #"Lasso": Lasso(alpha=0.01),#very slow
            "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5),
            "XGBRegressor(10,2)": XGBRegressor(n_estimators=10, max_depth=2, learning_rate=0.01),
            "XGBRegressor(10,5)": XGBRegressor(n_estimators=10, max_depth=5, learning_rate=0.01),
            "XGBRegressor(20,2)": XGBRegressor(n_estimators=20, max_depth=2, learning_rate=0.01),
            "XGBRegressor(20,5)": XGBRegressor(n_estimators=20, max_depth=5, learning_rate=0.01),
        }
        model_performance = sklearn_model_and_train(models, train_loader, vali_loader, use_wandb=args.wandb)
        performance_csv = pd.DataFrame.from_dict(model_performance, orient='index', columns=['CV_MSE'])
        performance_file = os.path.join(experiment_output_dir, f"sklearn_model_performance.csv")
        performance_csv.to_csv(performance_file)
        print(f"Model performance saved to: {performance_file}")
        #raise NotImplementedError("Sklearn model evaluation and submission creation not implemented yet.")
    else:
        # Define loss function, model, and perform training:
        loss_fn = nn.MSELoss()
        model = simple_model_and_train(train_loader, vali_loader, loss_fn, args.wandb)

        # Evaluate model on train, validation, and test data, create plots, and create final prediction submission
        # dataframe (with datetime annotation), and save it as submission file CSV:
        res_eval_train = simple_eval_and_submission_creation(
            train_loader,
            model,
            loss_fn,
            generate_timeseries_prediction=True,
            save_fig=os.path.join(experiment_output_dir, "plot_train.png"),
        )
        res_eval_vali = simple_eval_and_submission_creation(
            vali_loader,
            model,
            loss_fn,
            generate_timeseries_prediction=True,
            save_fig=os.path.join(experiment_output_dir, "plot_vali.png"),
        )

        if test_input_df is not None:
            res_eval_test_input = simple_eval_and_submission_creation(
                test_input_loader,
                model,
                loss_fn=None,  # no loss function for test set evaluation, because there is no ground truth in raw data
                generate_timeseries_prediction=True,
                save_fig=os.path.join(experiment_output_dir, "plot_test.png"),
                create_submission_df=test_start_datetime,
            )
            test_prediction_df = res_eval_test_input["ys_pred_df"]

            test_prediction_df_for_csv = test_prediction_df.copy()
            test_prediction_df_for_csv.index = test_prediction_df.index.tz_localize(tzinfo)
            test_prediction_df_for_csv.index = test_prediction_df.index.strftime(
                SUBMISSION_FILE_DATETIME_FORMAT
            )
            test_prediction_df_for_csv.index.name = "ID"
        
        # Log final evaluation metrics to wandb
        if args.wandb:
            wandb.log({
                "final_train_loss": res_eval_train["avg_loss"].item(),
                "final_validation_loss": res_eval_vali["avg_loss"].item(),
                "num_predictions": len(test_prediction_df),
            })
            
            # Log model configuration
            sample_batch, _ = next(iter(train_loader))
            wandb.config.update({
                "input_size": sample_batch.shape[-1] - 1,  # subtract 1 for timestamp
                "model_architecture": "SimpleAIFBOModel",
                "optimizer": "Adam",
                "learning_rate": 2.5e-4,
                "num_epochs": 200,
                "batch_size": 64,
                "loss_function": "MSELoss",
            })

        print(f"Experiment completed successfully!")
        print(f"Results saved to: {experiment_output_dir}")
        
        if test_input_df is not None:
            # write the submission file that can then be uploaded to the competition page:
            submission_file_path = os.path.join(experiment_output_dir, "submission_file.csv")
            test_prediction_df_for_csv.to_csv(
                submission_file_path,
                index=True,
                quoting=csv.QUOTE_ALL,
            )
            print(f"Submission file: {submission_file_path}")


def run_summary_mode(experiment_name, data_splits=None, summary_dir=None):
    """Run summary mode: load and summarize sklearn model performance across data splits."""
    print("Running in SUMMARY mode...")
    print(f"Summarizing results for experiment: {experiment_name}")
    
    # Get all available data splits if not specified
    if data_splits is None:
        all_splits = [d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
        data_splits = all_splits
        print(f"No specific data splits provided, using all available: {len(data_splits)} splits")
    else:
        print(f"Summarizing results for {len(data_splits)} specified data splits")
    
    # Collect performance data
    summary_data = []
    
    for data_split in data_splits:
        data_split_dir = os.path.join(OUTPUTS_DIR, data_split)
        experiment_dir = os.path.join(data_split_dir, "experiments", experiment_name)
        performance_file = os.path.join(experiment_dir, "sklearn_model_performance.csv")
        
        if not os.path.exists(performance_file):
            print(f"Warning: Performance file not found for data split '{data_split}' and experiment '{experiment_name}'")
            print(f"  Expected: {performance_file}")
            continue
            
        # Load performance data
        try:
            perf_df = pd.read_csv(performance_file, index_col=0)
            
            # Add data split info to each model's performance
            for model_name, row in perf_df.iterrows():
                summary_data.append({
                    'data_split': data_split,
                    'model': model_name,
                    'cv_mse': row['CV_MSE'],
                })
                
            print(f"✓ Loaded results for data split: {data_split}")
            
        except Exception as e:
            print(f"Error loading performance file for data split '{data_split}': {e}")
            continue
    
    if not summary_data:
        print("No performance data found! Make sure you have run sklearn experiments with the specified name.")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n" + "="*80)
    print(f"SKLEARN MODEL PERFORMANCE SUMMARY")
    print(f"Experiment Name: {experiment_name}")
    print(f"Number of Data Splits: {summary_df['data_split'].nunique()}")
    print(f"Number of Models: {summary_df['model'].nunique()}")
    print(f"="*80)
    
    # Overall summary statistics
    print(f"\n📊 OVERALL PERFORMANCE STATISTICS:")
    model_summary = summary_df.groupby('model')['cv_mse'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    model_summary.columns = ['Data Splits', 'Mean CV MSE', 'Std CV MSE', 'Min CV MSE', 'Max CV MSE']
    print(model_summary.sort_values('Mean CV MSE').to_string())
    
    # Best performing model per data split
    print(f"\n🏆 BEST MODEL PER DATA SPLIT:")
    best_per_split = summary_df.loc[summary_df.groupby('data_split')['cv_mse'].idxmin()][['data_split', 'model', 'cv_mse']]
    best_per_split = best_per_split.sort_values('cv_mse')
    print(best_per_split.to_string(index=False))
    
    # Model performance across data splits (pivot table)
    print(f"\n📈 MODEL PERFORMANCE MATRIX (CV MSE):")
    pivot_df = summary_df.pivot(index='data_split', columns='model', values='cv_mse').round(6)
    print(pivot_df.to_string())
    
    # Save summary to file only if summary_dir is specified
    if summary_dir is not None:
        os.makedirs(summary_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(summary_dir, f"sklearn_summary_{experiment_name}_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        pivot_file = os.path.join(summary_dir, f"sklearn_pivot_{experiment_name}_{timestamp}.csv")
        pivot_df.to_csv(pivot_file)
        
        print(f"\n💾 SUMMARY SAVED:")
        print(f"  Detailed summary: {summary_file}")
        print(f"  Performance matrix: {pivot_file}")
    else:
        print(f"\n💾 No summary directory specified (--summarydir), files not saved.")


if __name__ == "__main__":
    # Parse command line arguments with subparsers
    parser = argparse.ArgumentParser(description='ELIAS Bosch AI for Building Optimisation prediction')
    subparsers = parser.add_subparsers(dest='mode', help='Available modes', required=True)
    
    # Resample mode subparser
    resample_parser = subparsers.add_parser('resample', help='Load raw data and create resampled parquet files')
    resample_parser.add_argument('--train_start', type=str, required=True,
                                help='Training start date in YYYY-MM format')
    resample_parser.add_argument('--train_end', type=str, required=True,
                                help='Training end date in YYYY-MM format')  
    resample_parser.add_argument('--test_start', type=str, default=None,
                                help='Test start date in YYYY-MM format. If not provided, defaults to one month after training end date')
    resample_parser.add_argument('--test_end', type=str, default=None,
                                help='Test end date in YYYY-MM format. If not provided, defaults to two months after training end date')
    
    # Experiment mode subparser  
    experiment_parser = subparsers.add_parser('experiment', help='Run ML experiment on preprocessed data')
    experiment_parser.add_argument('--name', type=str, required=True, help='Experiment name identifier')
    
    # Experiment-specific parameters
    experiment_parser.add_argument('--device', type=str, default=None,
                                  help='Device to use for computation (default: auto-detect, options: cpu, cuda, cuda:0, etc.)')
    experiment_parser.add_argument('--sklearn', action='store_true',
                                  help='Use sklearn models instead of PyTorch model')
    experiment_parser.add_argument('--wandb', action='store_true',
                                  help='Enable wandb logging')
    experiment_parser.add_argument('--feature_hours', type=float, default=1.0,
                                  help='Number of past hours to use as features (default: 1)')
    experiment_parser.add_argument('--input_seq_step', type=int, default=1,
                                  help='Step size for input sequence (default: 1)')
    experiment_parser.add_argument('--stride', type=int, default=1,
                                  help='Stride for moving the input window (default: 1)')
    experiment_parser.add_argument('--use_custom_date', action='store_true',
                                  help='Use custom date features in the dataset')
    experiment_parser.add_argument('--use_custom_sensor', action='store_true',
                                  help='Use custom sensor features in the dataset')
    experiment_parser.add_argument('--data_splits', type=str, nargs='+', default=None,
                                  help='List of data split folder names to run experiments on (e.g., train_2022-01_to_2022-03_test_2022-04_to_2022-05)')
    
    # Summary mode subparser
    summary_parser = subparsers.add_parser('summary', help='Summarize sklearn model performance across data splits')
    summary_parser.add_argument('--name', type=str, required=True,
                               help='Experiment name identifier to summarize results for')
    summary_parser.add_argument('--data_splits', type=str, nargs='+', default=None,
                               help='List of specific data split folder names to include in summary (if not provided, includes all available)')
    summary_parser.add_argument('--summarydir', type=str, default=None,
                               help='Directory to export summary files to (if not provided, no files are saved)')
    
    args = parser.parse_args()
    
    # Validate date format for resample mode
    if args.mode == 'resample':
        import re
        date_pattern = r'^\d{4}-\d{2}$'
        for date_arg, date_value in [('train_start', args.train_start), ('train_end', args.train_end), 
                                   ('test_start', args.test_start), ('test_end', args.test_end)]:
            if date_arg in ['test_start', 'test_end'] and date_value is None:
                continue  # skip validation if not provided
            if not re.match(date_pattern, date_value):
                parser.error(f"Invalid date format for --{date_arg}: {date_value}. Expected format: YYYY-MM")
    
    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == 'resample':
        run_resample_mode(args.train_start, args.train_end, args.test_start, args.test_end)
        
    elif args.mode == 'summary':
        run_summary_mode(args.name, args.data_splits, args.summarydir)
        
    elif args.mode == 'experiment':
        # Setup device and torch configuration for experiment mode
        device = setup_device(args.device)
        torch.manual_seed(0)
        torch.set_default_device(device)
        print(f"Using device: {torch.get_default_device()}")
        torch.set_default_dtype(torch.float64)  # with lower than float64 precision, the eventual timestamps may be off
        
        # Setup wandb logging
        use_wandb = setup_wandb() and getattr(args, 'wandb', False)
        if use_wandb:
            wandb.init(
                project="kaggle-energy",
                name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.data_split}",
                config={
                    "device": str(torch.get_default_device()),
                    "target_variable": TARGET_VARIABLE_NAME,
                    "predictor_variables": EXAMPLE_PREDICTOR_VARIABLE_NAMES,
                    "num_predictor_variables": len(EXAMPLE_PREDICTOR_VARIABLE_NAMES),
                    "resample_freq_min": RESAMPLE_FREQ_MIN,
                    "eps": EPS,
                    "random_seed": 0,
                    "torch_dtype": str(torch.get_default_dtype()),
                    "data_dir": DATA_DIR,
                    "outputs_dir": OUTPUTS_DIR,
                    "data_split": args.data_split,
                    "feature_hours": getattr(args, 'feature_hours', 1),
                    "input_seq_step": getattr(args, 'input_seq_step', 1),
                    "stride": getattr(args, 'stride', 1),
                    "use_custom_date": getattr(args, 'use_custom_date', False),
                },
                tags=["pytorch", "timeseries", "building-optimization", "kaggle"]
            )
        all_splits = os.listdir(OUTPUTS_DIR)
        if args.data_splits is None:
            data_splits_to_run = all_splits
        else:
            for data_split in args.data_splits:
                if data_split not in all_splits:
                    raise ValueError(f"Data split folder {data_split} not found in outputs directory.")
            data_splits_to_run = args.data_splits

        # Run experiments
        for data_split in data_splits_to_run:
            print(f"Running experiment for data split: {data_split}")
            run_experiment_mode(data_split, args)
        
        # Finalize wandb logging
        if use_wandb:
            wandb.finish()

    print("Done.")