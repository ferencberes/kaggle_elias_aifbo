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
import holidays
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
OUTPUTS_DIR = "outputs"
TRAIN_DATA_FILE_PATHS = list(
    chain(
        #glob.glob(
        #    f"{DATA_DIR}/kaggle_dl/RBHU-2024-06/RBHU/**/*.parquet", recursive=True
        #),
        #glob.glob(
        #    f"{DATA_DIR}/kaggle_dl/RBHU-2024-07/RBHU/**/*.parquet", recursive=True
        #),
        #glob.glob(
        #    f"{DATA_DIR}/kaggle_dl/RBHU-2024-08/RBHU/**/*.parquet", recursive=True
        #),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-01/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-02/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-03/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-04/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-05/RBHU/**/*.parquet", recursive=True
        ),
    )
)
TEST_START_DATETIME = datetime(2025, 6, 1)  # start of test set
TEST_INPUT_DATA_FILE_PATHS = list(
    chain(
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-05/RBHU/**/*.parquet",
            recursive=True,
        ),  # just for the lag
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-06/RBHU/**/*.parquet", recursive=True
        ),
        glob.glob(
            f"{DATA_DIR}/kaggle_dl/RBHU-2025-07/RBHU/**/*.parquet", recursive=True
        ),
    )
)
RESAMPLE_FREQ_MIN = 10  # the frequency in minutes to resample the raw irregularly sampled timeseries to, using ffill
EPS = 1e-6
TARGET_VARIABLE_NAME = "B205WC000.AM02"  # the target variable to be predicted
EXAMPLE_PREDICTOR_VARIABLE_NAMES = [
    #ORIGINAL 2:
    "B205WC000.AM01",  # a supply temperature chilled water
    "B106WS01.AM54",  # an external temperature
    #high abs corr weather:
    #'B106WS01.AM51',  # light intensity
    #'B106WS01.AM53',  # humidity
    #abova 0.4 spcorr
    #'B205WC140.AC21',# PRIMARY VALVE 1
    #'B205HW010.PA11',# NUMBER OF STARTS
    #'B205HW020.PA11',# NUMBER OF STARTS
    #'B205WC001.AM71',# TOTAL VOLUME CHILLED WATER
    #'B205WC000.AM71',# VOLUME CHILLED WATER BP201/202/206
    # same num best lasso weights:
    #'B205WC140.AC21',# PRIMARY VALVE 1
    #'B205WC030.AM55_3',# ACTUAL CAPACITY
    #'B201AH162.AC21',# COOLER VALVE
    ##'B205WC002.RA001',# SPEED CHILLED WATER PUMP
    #'B205WC001.DM82_1',# FAULT DIFF-PRESSURE FILTER 2
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
        plt.savefig(f"{OUTPUTS_DIR}/input_data_sample_timeseries_plot.png", bbox_inches="tight")
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
    full_multivariate_timeseries_df, add_dummy_y=False, normalize=False, feature_hours=1, input_seq_step=1, stride=1, use_custom_date_features=False
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

    input_seq_len = int(60 / RESAMPLE_FREQ_MIN) * feature_hours  # hours
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

    column_names = timeseries_df.columns
    print('Dataset columns:', column_names.tolist())

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
        # next previous datetime features:
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

        # finally: numerical (not date related) predictor variables:
        for predictor in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
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

def check_preprocessed_files_exist():
    """Check if both preprocessed files exist in the outputs directory.
    
    Returns:
        bool: True if both files exist, False otherwise
    """
    train_file = f"{OUTPUTS_DIR}/preproc_full_train_df.parquet"
    test_file = f"{OUTPUTS_DIR}/preproc_test_input_df.parquet"
    
    train_exists = os.path.exists(train_file)
    test_exists = os.path.exists(test_file)
    
    if train_exists and test_exists:
        print("Both preprocessed files found:")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        print("Skipping raw file processing and loading preprocessed data directly.")
        return True
    else:
        if not train_exists:
            print(f"Preprocessed train file not found: {train_file}")
        if not test_exists:
            print(f"Preprocessed test file not found: {test_file}")
        print("Will process raw files...")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ELIAS Bosch AI for Building Optimisation prediction')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for computation (default: auto-detect, options: cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--sklearn', action='store_true',
                       help='Use sklearn models instead of PyTorch model')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--feature_hours', type=int, default=1,
                       help='Number of past hours to use as features (default: 1)')
    parser.add_argument('--input_seq_step', type=int, default=1,
                       help='Step size for input sequence (default: 1)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for moving the input window (default: 1)')
    parser.add_argument('--use_custom_date', action='store_true',
                       help='Use custom date features in the dataset')
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    
    # Setup device and torch configuration
    device = setup_device(args.device)

    torch.manual_seed(0)
    torch.set_default_device(device)
    print(f"Using device: {torch.get_default_device()}")
    torch.set_default_dtype(
        torch.float64
    )  # with lower than float64 precision, the eventual timestamps may be off
    
    # Setup wandb logging
    use_wandb = setup_wandb() and args.wandb
    #print(f"Using wandb logging: {use_wandb}")
    if use_wandb:
        wandb.init(
            project="kaggle-energy",
            name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
                "test_start_datetime": TEST_START_DATETIME.isoformat(),
                #"submission_file_path": SUBMISSION_FILE_PATH,
            },
            tags=["pytorch", "timeseries", "building-optimization", "kaggle"]
        )
    
    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Check if preprocessed files already exist
    preprocessed_files_exist = check_preprocessed_files_exist()
    
    if preprocessed_files_exist:
        # Load preprocessed data directly
        print("Loading preprocessed train data...")
        full_train_df = pd.read_parquet(f"{OUTPUTS_DIR}/preproc_full_train_df.parquet")
        
        print("Loading preprocessed test data...")
        test_input_df = pd.read_parquet(f"{OUTPUTS_DIR}/preproc_test_input_df.parquet")
        
        print("Preprocessed data loaded successfully.")
    else:
        # Process raw data as usual
        print("Processing raw data files...")
        full_train_df = simple_load_and_resample_data(
            TRAIN_DATA_FILE_PATHS,
            generate_sample_plots=[TARGET_VARIABLE_NAME] + EXAMPLE_PREDICTOR_VARIABLE_NAMES,
            save_load_df=f"{OUTPUTS_DIR}/preproc_full_train_df.parquet",
        )
        test_input_df = simple_load_and_resample_data(
            TEST_INPUT_DATA_FILE_PATHS,
            save_load_df=f"{OUTPUTS_DIR}/preproc_test_input_df.parquet",
        )
    
    tzinfo = full_train_df.index.tzinfo

    # Turn it into torch datasets for simple prediction from past to future, with simple features:
    full_train_dataset, full_train_dataset_info = simple_feature_dataset(
        full_train_df, add_dummy_y=False, normalize=True, feature_hours=args.feature_hours, 
        input_seq_step=args.input_seq_step, stride=args.stride, use_custom_date_features=args.use_custom_date
    )
    #stride is set to 1 for test dataset, because we want predictions for every time step in test set
    test_input_dataset, _ = simple_feature_dataset(
        test_input_df, add_dummy_y=True, normalize=full_train_dataset_info, feature_hours=args.feature_hours, 
        input_seq_step=args.input_seq_step, stride=1, use_custom_date_features=args.use_custom_date
    )

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
    test_input_loader = DataLoader(test_input_dataset, batch_size=64, shuffle=False)

    if args.sklearn:
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from xgboost import XGBRegressor

        # Define and train sklearn models
        models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(alpha=0.01),
            "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5),
            "XGBRegressor": XGBRegressor(n_estimators=10, max_depth=2, learning_rate=0.01),
        }
        model_performance = sklearn_model_and_train(models, train_loader, vali_loader, use_wandb=use_wandb)
        performance_csv = pd.DataFrame.from_dict(model_performance, orient='index', columns=['CV_MSE'])
        name = '' if args.name is None else '_' + args.name
        performance_csv.to_csv(f"{OUTPUTS_DIR}/sklearn_model_performance{name}.csv")
        raise NotImplementedError("Sklearn model evaluation and submission creation not implemented yet.")
    else:
        # Define loss function, model, and perform training:
        loss_fn = nn.MSELoss()
        model = simple_model_and_train(train_loader, vali_loader, loss_fn, use_wandb)

        # Evaluate model on train, validation, and test data, create plots, and create final prediction submission
        # dataframe (with datetime annotation), and save it as submission file CSV:
        res_eval_train = simple_eval_and_submission_creation(
            train_loader,
            model,
            loss_fn,
            generate_timeseries_prediction=True,
            save_fig=f"{OUTPUTS_DIR}/plot_train.png",
        )
        res_eval_vali = simple_eval_and_submission_creation(
            vali_loader,
            model,
            loss_fn,
            generate_timeseries_prediction=True,
            save_fig=f"{OUTPUTS_DIR}/plot_vali.png",
        )
        res_eval_test_input = simple_eval_and_submission_creation(
            test_input_loader,
            model,
            loss_fn=None,  # no loss function for test set evaluation, because there is no ground truth in raw data
            generate_timeseries_prediction=True,
            save_fig=f"{OUTPUTS_DIR}/plot_test.png",
            create_submission_df=TEST_START_DATETIME,
        )
        test_prediction_df = res_eval_test_input["ys_pred_df"]

    test_prediction_df_for_csv = test_prediction_df.copy()
    test_prediction_df_for_csv.index = test_prediction_df.index.tz_localize(tzinfo)
    test_prediction_df_for_csv.index = test_prediction_df.index.strftime(
        SUBMISSION_FILE_DATETIME_FORMAT
    )
    test_prediction_df_for_csv.index.name = "ID"
    
    # Log final evaluation metrics to wandb
    if use_wandb:
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
    
    # write the submission file that can then be uploaded to the competition page:
    test_prediction_df_for_csv.to_csv(
        SUBMISSION_FILE_PATH,
        index=True,
        quoting=csv.QUOTE_ALL,
    )
    
    # Finalize wandb logging
    if use_wandb:
        wandb.finish()

    print("Done.")