"""This is a script that can serve as an initialization for participating in the
ELIAS Bosch AI for Building Optimisation prediction competition.

It contains simple versions of the essential components for
* loading and preprocessing data,
* defining a simple torch pairs dataset for causal prediction (i.e., using only past to predict future),
* defining a simple toy example model and training it,
* and evaluating and creating the submission file (`submission_file.csv`) of that model on the test input data.

Note that all of the components are just starting points, and many aspects can still be improved, see also `README.md`.
"""

# Copyright (c) 2025 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import csv
from datetime import datetime
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

from typing import List

torch.manual_seed(0)
torch.set_default_device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.get_default_device())
torch.set_default_dtype(
    torch.float64
)  # with lower than float64 precision, the eventual timestamps may be off

data_dir = "data/kaggle_dl"
metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
feature_information = pd.read_csv('test_set_feature_information.csv', index_col=0)
potential_features = feature_information[
    (feature_information['missing_ratio'] < 0.2) &
    (feature_information['nunique_count'] > 2)
].copy()
potential_features = potential_features.index.tolist()
# keep only features that are available during the test set period
metadata = metadata[metadata['object_id'].isin(potential_features)].copy()

DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
TRAIN_DATA_FILE_PATHS = list(
    chain(
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
    "B205WC000.AM01",  # a supply temperature chilled water
    "B106WS01.AM54",  # an external temperature
    #extra features
    # made it worse
    #"B205WC001.AM01", # a supply temperature chilled water - it looks to be too sparse during the test period
    #"B205WC002.RA001", # SPEED CHILLED WATER PUMP
]  # example predictor variables

#TODO:
#B205WC100.PA72, #VOLUME FEEDING HOT WATER SYSTEM

from feature_groups import get_cooler_valves, get_active_setpoints, get_co2_concentrations, get_humidity_sensors, get_controller_building_sensors, get_room_temperatures

rerun_all = False

use_cooler_valves = True
use_active_setpoints = False
use_fc_room_temps = True
use_rc_room_temps = False

use_co2_concentrations = False
use_humidity_sensors = False
use_controller_building_sensors = False

if use_cooler_valves:
    cooler_valves = get_cooler_valves(metadata, enable_rooms=True)
    cooler_valves_ids = cooler_valves['object_id'].unique().tolist()
    print(f"Using {len(cooler_valves_ids)} cooler valves as predictor variables.")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += cooler_valves_ids

#room_sizes = metadata[['room', 'bim_room_area']].dropna().drop_duplicates(subset=['room'])
#topk_rooms = room_sizes.nlargest(50, 'bim_room_area')['room'].tolist()
#metadata = metadata[metadata['room'].isin(topk_rooms)].copy()

if use_active_setpoints:
    active_setpoints = get_active_setpoints(metadata).drop_duplicates(subset=['object_id'])
    setpoints_by_room = active_setpoints.groupby('room')['object_id'].apply(list)
    active_setpoints_ids = active_setpoints['object_id'].unique().tolist()
    print(f"Using {len(active_setpoints_ids)} active setpoints as predictor variables.")
    print(f"Number of rooms with active setpoints: {len(setpoints_by_room)}")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += active_setpoints_ids

if use_fc_room_temps:
    fc_room_temps = get_room_temperatures(metadata, class_id='FC')
    fc_room_temp_ids = fc_room_temps['object_id'].unique().tolist()
    fc_temp_by_room = fc_room_temps.groupby('room')['object_id'].apply(list)
    print(f"Using {len(fc_room_temp_ids)} FC room temperature sensors as predictor variables.")
    print(f"Number of rooms with FC room temperature sensors: {len(fc_temp_by_room)}")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += fc_room_temp_ids

if use_rc_room_temps:
    rc_room_temps = get_room_temperatures(metadata, class_id='RC')
    rc_room_temp_ids = rc_room_temps['object_id'].unique().tolist()
    rc_temp_by_room = rc_room_temps.groupby('room')['object_id'].apply(list)
    print(f"Using {len(rc_room_temp_ids)} RC room temperature sensors as predictor variables.")
    print(f"Number of rooms with RC room temperature sensors: {len(rc_temp_by_room)}")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += rc_room_temp_ids

if use_co2_concentrations:
    co2_concentration_sensors = get_co2_concentrations(metadata).copy()
    co2_concentration_sensors['room'] = co2_concentration_sensors['room'].fillna('NoRoom')
    #print(co2_concentration_sensors.isnull().mean().sort_values(ascending=False).head(10))
    co2_concentration_by_room = co2_concentration_sensors.groupby('room')['object_id'].apply(list)
    co2_concentration_ids = co2_concentration_sensors['object_id'].unique().tolist()
    print(f"Using {len(co2_concentration_ids)} CO2 concentration sensors as predictor variables.")
    print(f"Number of rooms with CO2 concentration sensors: {len(co2_concentration_by_room)}")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += co2_concentration_ids

if use_humidity_sensors:
    humidity_sensors = get_humidity_sensors(metadata)
    #TODO: there is high missing room rate (25 room out of almost 100 sensors)
    #humidity_sensors['room'] = humidity_sensors['room'].fillna('NoRoom')
    #humidity_by_room = humidity_sensors.groupby('room')['object_id'].apply(list)
    humidity_sensor_ids = humidity_sensors['object_id'].unique().tolist()
    print(f"Using {len(humidity_sensor_ids)} humidity sensors as predictor variables.")
    #print(f"Number of rooms with humidity sensors: {len(humidity_by_room)}")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += humidity_sensor_ids

if use_controller_building_sensors:
    controller_building_sensors = get_controller_building_sensors(metadata, building_id='B205')
    controller_building_sensor_ids = controller_building_sensors['object_id'].unique().tolist()
    #removing already used predictor variables
    controller_building_sensor_ids = list(set(controller_building_sensor_ids) - set(EXAMPLE_PREDICTOR_VARIABLE_NAMES))
    if TARGET_VARIABLE_NAME in controller_building_sensor_ids:
        controller_building_sensor_ids.remove(TARGET_VARIABLE_NAME)
    print(f"Using {len(controller_building_sensor_ids)} controller building B205 sensors as predictor variables.")
    EXAMPLE_PREDICTOR_VARIABLE_NAMES += controller_building_sensor_ids

print("Predictor variables:")
#print(EXAMPLE_PREDICTOR_VARIABLE_NAMES)
print(f"Using {len(EXAMPLE_PREDICTOR_VARIABLE_NAMES)} predictor variables.")
print('Do you want to proceed? (y/n)')
proceed = input()
if proceed.lower() != 'y':
    print("Exiting.")
    exit()

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
        )
        plt.title("Input data timeseries")
        for i, col in enumerate(generate_sample_plots):
            axs[i].plot(multivariate_timeseries_df[col], label=col, linewidth=0.75)
            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].legend(fontsize="small")
        plt.savefig(f"{OUTPUTS_DIR}/input_data_sample_timeseries_plot.png")
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
                plt.savefig(save_fig)
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
    full_multivariate_timeseries_df, add_dummy_y=False, normalize=False, inspect_nans=True
):
    """Create a torch dataset from the multivariate timeseries dataframe, intended for causal prediction (just use past
    to predict future), consisting of samples of predictor features
    (past sequence up to time step t and daytime features) as well as target variable (value of to-be-predicted
    variable/timeseries, at step t plus forecast ahead), each annotatetd with the timestamp of the prediction
    (i.e., the time at which the target variable is to be predicted).

    Important: to be adapted for actual models for competition.

    Args:
        multivariate_timeseries_df: The multivariate timeseries dataframe of measurements.
        add_dummy_y: If True, the to-be-predicted target variable will be set to NaN. This option can be used to create
            a dataset also for the test input data, where no target variable values are available, but nonetheless
            a column needs to exist for the target variable.
            If False, the target variable will be used as is, i.e., the dataset can be used for training or validation
            where the target variable is available in the raw data.
        normalize: normalize selected columns of the timeseries data.
            if True, take mean and std from the data (and return that info),
            if a dict, use the contained mean and std.
    Returns:
        A torch dataset containing pairs of input features and target variable values. Note that both, input features'
        and target variable's first entry is the timestamp of the prediction, i.e., the time at which the target
        variable is to be predicted.
    """

    info = {}

    input_seq_len = int(60 / RESAMPLE_FREQ_MIN) * 1  # hours
    input_seq_step = 1
    stride = 1  # step size for sliding window
    predict_ahead = int(60 / RESAMPLE_FREQ_MIN) * 3  # hours

    # restrict to only relevant/valid data:
    timeseries_df = full_multivariate_timeseries_df[
        [
            col
            for col in EXAMPLE_PREDICTOR_VARIABLE_NAMES + [TARGET_VARIABLE_NAME]
            if col in full_multivariate_timeseries_df.columns
        ]
    ]

    if inspect_nans:
        first_valid_idx = timeseries_df.notna().all(axis=1).idxmax()
        timeseries_df = timeseries_df.loc[first_valid_idx:].copy()
    else:
        timeseries_df = timeseries_df.ffill().bfill().copy()

    if use_active_setpoints:
        for room, sp_ids in setpoints_by_room.items():
            sensors_available = list(
                set(sp_ids).intersection(set(timeseries_df.columns))
            )
            if len(sensors_available) == 0:
                continue
            room_col = 'SumActiveSetpoint_' + room
            timeseries_df[room_col] = timeseries_df[sensors_available].sum(axis=1)
            if not room_col in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
                EXAMPLE_PREDICTOR_VARIABLE_NAMES.append(room_col)

    if use_fc_room_temps:
        for room, temp_ids in fc_temp_by_room.items():
            sensors_available = list(
                set(temp_ids).intersection(set(timeseries_df.columns))
            )
            if len(sensors_available) == 0:
                continue
            room_col = 'AvgFCRoomTemp_' + room
            timeseries_df[room_col] = timeseries_df[sensors_available].mean(axis=1)
            if not room_col in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
                EXAMPLE_PREDICTOR_VARIABLE_NAMES.append(room_col)

    if use_rc_room_temps:
        for room, temp_ids in rc_temp_by_room.items():
            sensors_available = list(
                set(temp_ids).intersection(set(timeseries_df.columns))
            )
            if len(sensors_available) == 0:
                continue
            room_col = 'AvgRCRoomTemp_' + room
            timeseries_df[room_col] = timeseries_df[sensors_available].mean(axis=1)
            if not room_col in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
                EXAMPLE_PREDICTOR_VARIABLE_NAMES.append(room_col)

    if use_co2_concentrations:
        for room, co2_ids in co2_concentration_by_room.items():
            sensors_available = list(
                set(co2_ids).intersection(set(timeseries_df.columns))
            )
            if len(sensors_available) == 0:
                continue
            room_col = 'AvgCO2Concentration_' + room
            timeseries_df[room_col] = timeseries_df[sensors_available].mean(axis=1)
            if not room_col in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
                EXAMPLE_PREDICTOR_VARIABLE_NAMES.append(room_col)

    """
    if use_humidity_sensors:
        for room, hum_ids in humidity_by_room.items():
            sensors_available = list(
                set(hum_ids).intersection(set(timeseries_df.columns))
            )
            if len(sensors_available) == 0:
                continue
            room_col = 'AvgHumidity_' + room
            timeseries_df[room_col] = timeseries_df[sensors_available].mean(axis=1)
            if not room_col in EXAMPLE_PREDICTOR_VARIABLE_NAMES:
                EXAMPLE_PREDICTOR_VARIABLE_NAMES.append(room_col)
    """

    if add_dummy_y:
        timeseries_df[TARGET_VARIABLE_NAME] = np.nan

    # extract and add some faetures:
    datetime_list = timeseries_df.index.to_pydatetime()
    timeseries_df["timestamp"] = [dtime.timestamp() for dtime in datetime_list]
    timeseries_df["minute_of_day"] = (
        timeseries_df.index.hour * 60 + timeseries_df.index.minute
    )
    timeseries_df["day_of_week"] = timeseries_df.index.dayofweek
    """#due to the bug this feature had no meaning, commenting out for now
    timeseries_df["day_of_year"] = timeseries_df.index.dayofweek
    timeseries_df["yeartime_sin"] = np.sin(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    timeseries_df["yeartime_cos"] = np.cos(
        2 * np.pi * timeseries_df["day_of_year"] / 365
    )
    """

    timeseries_df["daytime_sin"] = np.sin(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )
    timeseries_df["daytime_cos"] = np.cos(
        2 * np.pi * timeseries_df["minute_of_day"] / (24 * 60)
    )

    column_names = timeseries_df.columns

    #print(column_names.value_counts())

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

    pbar = tqdm(total=(data.shape[0] - input_seq_len - predict_ahead) // stride)
    for i in range(0, data.shape[0] - input_seq_len - predict_ahead, stride):
        selected_features = []
        
        timestamp = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("timestamp")
        ].unsqueeze(0)
        selected_features.append(timestamp)

        daytime_sin = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("daytime_sin")
        ].unsqueeze(0)
        selected_features.append(daytime_sin)

        daytime_cos = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("daytime_cos")
        ].unsqueeze(0)
        selected_features.append(daytime_cos)

        day_of_week = torch.nn.functional.one_hot(
            data[
                i + input_seq_len + predict_ahead, column_names.get_loc("day_of_week")
            ].to(dtype=torch.long),
            num_classes=7,
        )
        selected_features.append(day_of_week)
        """
        yeartime_sin = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("yeartime_sin")
        ].unsqueeze(0)
        yeartime_cos = data[
            i + input_seq_len + predict_ahead, column_names.get_loc("yeartime_cos")
        ].unsqueeze(0)
        
        X.append(
            torch.cat(
                [
                    timestamp,
                    example_predictor_variable_0,
                    example_predictor_variable_1,
                    day_of_week,
                    #yeartime_sin,
                    #yeartime_cos,
                    daytime_sin,
                    daytime_cos,
                ]
            )
        )
        """

        for j in range(len(EXAMPLE_PREDICTOR_VARIABLE_NAMES)):
            col_name = EXAMPLE_PREDICTOR_VARIABLE_NAMES[j]

            if use_active_setpoints:
                if col_name in active_setpoints_ids:
                    continue # skip raw setpoint values, use only room aggregated ones

            if use_fc_room_temps:
                if col_name in fc_room_temp_ids:
                    continue # skip raw fc room temp sensor values, use only room aggregated ones

            if use_rc_room_temps:
                if col_name in rc_room_temp_ids:
                    continue # skip raw rc room temp sensor values, use only room aggregated ones

            if use_co2_concentrations:
                if col_name in co2_concentration_ids:
                    continue # skip raw co2 concentration values, use only room aggregated ones

            """
            if use_humidity_sensors:
                if col_name in humidity_sensor_ids:
                    continue # skip raw humidity sensor values, use only room aggregated ones
            """
            
            if col_name in column_names:
                example_predictor_variable = normalization_fn(
                    data[
                        i : i + input_seq_len : input_seq_step,
                        column_names.get_loc(col_name),
                    ],
                    col_name,
                )
            else:
                example_predictor_variable = torch.zeros(
                    input_seq_len // input_seq_step,
                    dtype=torch.get_default_dtype(),
                )
            selected_features.append(example_predictor_variable)

        X.append(torch.cat(selected_features))

        target_variable = data[
            i + input_seq_len + predict_ahead,
            column_names.get_loc(TARGET_VARIABLE_NAME),
        ].unsqueeze(0)

        Y.append(torch.cat([timestamp, target_variable]))

        pbar.update(1)
    pbar.close()

    X = torch.stack(X)
    Y = torch.stack(Y)

    dataset = TensorDataset(X, Y)

    return dataset, info


def simple_model_and_train(train_loader, vali_loader, loss_fn):
    """Define a simple prediction model and train it on the given training data loader.

    Important: to be adapted for actual models for competition.
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
        
    class MultiChannelAIFBOModel(nn.Module):
        def __init__(self, input_size:int, hidden_other:int, hidden_by_channels:List[int], predictors_by_channels:List[list]):
            super().__init__()
            assert len(hidden_by_channels) == len(predictors_by_channels), "Length of hidden_by_channels must match length of predictors_by_channels"
            print(f"Input size: {input_size}") # timestamp has already been excluded from input_size
            input_seq_len = int(60 / RESAMPLE_FREQ_MIN) * 1  # hours
            self.channel_group_sizes = [len(ids) * input_seq_len for ids in predictors_by_channels]
            self.other_input_size = input_size - sum(self.channel_group_sizes)
            print(f"Other input size: {self.other_input_size}")
            self.mlp_other = torchvision.ops.MLP(
                in_channels=self.other_input_size,
                hidden_channels=[hidden_other, hidden_other],
                norm_layer=nn.LayerNorm,
            ).to(dtype=torch.get_default_dtype())
            self.mlp_by_channels = nn.ModuleList()
            for channel_size, hidden_size in zip(self.channel_group_sizes, hidden_by_channels):
                mlp_channel = torchvision.ops.MLP(
                    in_channels=channel_size,
                    hidden_channels=[hidden_size, hidden_size],
                    norm_layer=nn.LayerNorm,
                ).to(dtype=torch.get_default_dtype())
                self.mlp_by_channels.append(mlp_channel)
            self.final_mlp = torchvision.ops.MLP(
                in_channels=hidden_other + sum(hidden_by_channels),
                hidden_channels=[128, 1],
                norm_layer=nn.LayerNorm,
            ).to(dtype=torch.get_default_dtype())

        def forward(self, x):
            timestamp_of_prediction = x[:, :1]
            x_other = x[:, 1 : 1 + self.other_input_size]# a few datetime features + other predictors
            x_channels = x[:, 1 + self.other_input_size :]# channel group predictors
            y_other = self.mlp_other(x_other)
            y_by_channels = []
            start_idx = 0
            for mlp_channel, channel_size in zip(self.mlp_by_channels, self.channel_group_sizes):
                x_channel = x_channels[:, start_idx : start_idx + channel_size]
                y_channel = mlp_channel(x_channel)
                y_by_channels.append(y_channel)
                start_idx += channel_size
            y_core = self.final_mlp(torch.cat([y_other] + y_by_channels, dim=1))
            return torch.cat([timestamp_of_prediction, y_core], dim=1)

    x, _ = next(iter(train_loader))
    input_size = (
        x.shape[-1] - 1
    )  # Get the input size from the first batch, subtract 1 for the timestamp

    if use_cooler_valves or use_active_setpoints or use_fc_room_temps or use_rc_room_temps or use_co2_concentrations or use_humidity_sensors or use_controller_building_sensors:
        hidden_other = 128
        predictors_by_channels, hidden_by_channels = [], []
        if use_cooler_valves:
            predictors_by_channels.append(cooler_valves_ids)
            hidden_by_channels.append(64)
        if use_active_setpoints:
            predictors_by_channels.append(setpoints_by_room.index.tolist())
            hidden_by_channels.append(64)
        if use_fc_room_temps:
            predictors_by_channels.append(fc_temp_by_room.index.tolist())
            hidden_by_channels.append(64)
        if use_rc_room_temps:
            predictors_by_channels.append(rc_temp_by_room.index.tolist())
            hidden_by_channels.append(64)
        if use_co2_concentrations:
            predictors_by_channels.append(co2_concentration_by_room.index.tolist())
            hidden_by_channels.append(64)
        if use_humidity_sensors:
            #predictors_by_channels.append(humidity_by_room.index.tolist())
            predictors_by_channels.append(humidity_sensor_ids)
            hidden_by_channels.append(32)#64)
        if use_controller_building_sensors:
            predictors_by_channels.append(controller_building_sensor_ids)
            hidden_by_channels.append(64)

        model = MultiChannelAIFBOModel(input_size=input_size, hidden_other=hidden_other, hidden_by_channels=hidden_by_channels, predictors_by_channels=predictors_by_channels)
    else:
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

    return model


if __name__ == "__main__":
    reload_prepared_pt_files = (not rerun_all)
    
    # Load raw data and prepare it into multivariate dataframes, and create dir for later outputs:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # name was shortened after C02 concentration update (AM22 channel included)
    prepared_data_dir = f"{OUTPUTS_DIR}/useCoolerV_{use_cooler_valves}_useActiveSp_{use_active_setpoints}_useCO2_{use_co2_concentrations}_useHumidity_{use_humidity_sensors}_useCtrlBldg_{use_controller_building_sensors}_useFCRoomT_{use_fc_room_temps}_useRCRoomT_{use_rc_room_temps}"
    os.makedirs(prepared_data_dir, exist_ok=True)
    full_train_dataset_path = f"{prepared_data_dir}/full_train_dataset.pt"
    test_input_dataset_path = f"{prepared_data_dir}/test_input_dataset.pt"
    tzinfo_path = f"{prepared_data_dir}/tzinfo.pkl"

    if reload_prepared_pt_files and os.path.exists(full_train_dataset_path) and os.path.exists(test_input_dataset_path):
        print("Loading prepared dataset files ...")
        full_train_dataset = torch.load(full_train_dataset_path)
        test_input_dataset = torch.load(test_input_dataset_path)
        with open(tzinfo_path, "rb") as f:
            tzinfo = pickle.load(f)
    else:
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
        with open(tzinfo_path, "wb") as f:
            pickle.dump(tzinfo, f)

        # Turn it into torch datasets for simple prediction from past to future, with simple features:
        full_train_dataset, full_train_dataset_info = simple_feature_dataset(
            full_train_df, add_dummy_y=False, normalize=True, inspect_nans=True
        )
        torch.save(full_train_dataset, full_train_dataset_path)
        
        test_input_dataset, _ = simple_feature_dataset(
            test_input_df, add_dummy_y=True, normalize=full_train_dataset_info, inspect_nans=False
        )
        torch.save(test_input_dataset, test_input_dataset_path)

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

    # Define loss function, model, and perform training:
    loss_fn = nn.MSELoss()
    model = simple_model_and_train(train_loader, vali_loader, loss_fn)

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
    
    # write the submission file that can then be uploaded to the competition page:
    test_prediction_df_for_csv.to_csv(
        SUBMISSION_FILE_PATH,
        index=True,
        quoting=csv.QUOTE_ALL,
    )

    print("Done.")