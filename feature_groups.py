import pandas as pd
import os

external_weather_measurements = [
    'B106WS01.AM50',  # light direction - follows daily / annual pattern
    'B106WS01.AM51',  # light intensity - follows daily / annual pattern
    'B106WS01.AM52',  # air pressure
    'B106WS01.AM53',  # humidity - could be relevant
    'B106WS01.AM54',  # external temperature
    'B106WS01.AM54_1',  # outdoor air 30min (AUL-TEMP)
    'B106WS01.AM54_2',  # outdoor air 24h (AUL-TEMP)
    'B106WS01.AM55',  # precipitation - too sparse
    'B106WS01.AM56',  # wind direction
    'B106WS01.AM57',  # wind speed
]

def get_cooler_valves(metadata, enable_rooms=True):
    """
    Get cooler valve sensors from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        enable_rooms (bool): If True, include valves assigned to rooms; if False, exclude them.
    Returns:
        pd.DataFrame: DataFrame of cooler valve sensors.
    """
    cooler_valves = metadata[metadata['channel']=='AC21']
    if enable_rooms:
        return cooler_valves
    else:
        return cooler_valves[cooler_valves['room'].isnull()]
    
def get_active_setpoints(metadata, k=None):
    """
    Get active setpoint sensors from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        k (int, optional): Number of top sensors to return based on room area. If None, return all.
    Returns:
        pd.DataFrame: DataFrame of active setpoint sensors.
    """
    active_setpoints = metadata[metadata['channel']=='VT03_2']
    if k is not None:
        active_setpoints = active_setpoints.nlargest(k, 'bim_room_area')
    return active_setpoints

def get_co2_concentrations(metadata, k=None):
    """
    Get CO2 concentration sensors from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        k (int, optional): Number of top sensors to return based on room area. If None, return all.
    Returns:
        pd.DataFrame: DataFrame of CO2 concentration sensors.
    """
    co2_sensors = metadata[(metadata['channel']=='AM21') & (metadata['dimension_text'].str.lower()=='ppm')]
    #sometimes CO2 concentration has AM22 channels and much more (AM23, AM24 etc. !!!) as well.. so I have more sensors but no extra room attributed to this change
    #co2_sensors = metadata[(metadata['channel'].isin(['AM21', 'AM22'])) & (metadata['dimension_text'].str.lower()=='ppm')]
    # it made results worse!!! when using AM21 and AM22
    if k is not None:
        co2_sensors = co2_sensors.nlargest(k, 'bim_room_area')
    return co2_sensors

def get_humidity_sensors(metadata, k=None):
    """
    Get humidity sensors from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        k (int, optional): Number of top sensors to return based on room area. If None, return all.
    Returns:
        pd.DataFrame: DataFrame of humidity sensors.
    """
    humidity_sensors = metadata[metadata['channel'].isin(['AM45', 'AM45_1', 'AM51'])]
    excluded_ids = [
        'B106WS01.AM51', #light intensity sensor
    ]
    humidity_sensors = humidity_sensors[humidity_sensors['object_id'].isin(excluded_ids) == False]
    if k is not None:
        humidity_sensors = humidity_sensors.nlargest(k, 'bim_room_area')
    return humidity_sensors

def get_controller_building_sensors(metadata, building_id='B205', excluded_channels=['AC21', 'VT03_2', 'AM21', 'AM45', 'AM45_1', 'AM51']):
    """
    Get sensors from a specific building, excluding certain channels.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        building_id (str): Building identifier to filter sensors.
        excluded_channels (list): List of channels to exclude.
    Returns:
        pd.DataFrame: DataFrame of sensors from the specified building controller.
    """
    building_sensors = metadata[metadata['object_id'].str.startswith(building_id)] 
    if excluded_channels:
        building_sensors = building_sensors[~building_sensors['channel'].isin(excluded_channels)]
    return building_sensors

def get_room_temperatures(metadata, class_id='FC', k=None):
    """
    Get AM01 room temperature sensors for a specific class (FC or RC).
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        class_id (str): Class identifier ('FC' or 'RC') to filter sensors.
        k (int, optional): Number of top sensors to return based on room area. If None, return all.
    Returns:
        pd.DataFrame: DataFrame of room temperature sensors for the specified class.
    """
    fancoil_temps = metadata[(metadata['object_id'].str.startswith('B201'+class_id)) & (metadata['description'].str.upper().str.contains('ROOM TEMPERATURE')) & (metadata['channel']=='AM01')]
    if k is not None:
        fancoil_temps = fancoil_temps.nlargest(k, 'bim_room_area')
    return fancoil_temps

def get_channel_sensors_with_description(metadata, channel, description_pattern, k=None):
    """
    Get sensors from a specific channel with descriptions starting with a given pattern.

    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        channel (str): Channel identifier to filter sensors.
        description_pattern (str): Description pattern to filter sensors.
        k (int, optional): Number of top sensors to return based on room area. If None, return all.
    Returns:
        pd.DataFrame: DataFrame of sensors from the specified channel with matching descriptions.
    """
    channel_sensors = metadata[(metadata['channel']==channel) & (metadata['description'].str.upper().str.startswith(description_pattern.upper()))]
    if k is not None:
        channel_sensors = channel_sensors.nlargest(k, 'bim_room_area')
    return channel_sensors

def select_potential_sensors_for_test_period(metadata, missing_ratio_threshold=0.2, nunique_count_threshold=2):
    """
    Select potential sensors for the test period based on missing ratio and unique count thresholds.
    Args:
        metadata (pd.DataFrame): Metadata DataFrame containing sensor information.
        missing_ratio_threshold (float): Maximum allowed missing ratio for sensors.
        nunique_count_threshold (int): Minimum required unique count for sensors.
    Returns:
        pd.DataFrame: Filtered metadata DataFrame with potential sensors.
    """
    #this csv file is generated during data preprocessing and saved to disk
    feature_information = pd.read_csv('test_set_feature_information.csv', index_col=0)
    potential_features = feature_information[
        (feature_information['missing_ratio'] < missing_ratio_threshold) &
        (feature_information['nunique_count'] > nunique_count_threshold)
    ].copy()
    potential_features = potential_features.index.tolist()
    metadata = metadata[metadata['object_id'].isin(potential_features)].copy()
    return metadata

def prepare_predictor_variables(data_dir:str, target_variable_name:str, interactive:bool=True, use_cooler_valves:bool=True, use_active_setpoints:bool=False, use_fc_room_temps:bool=False, use_rc_room_temps:bool=False, use_co2_concentrations:bool=False, use_humidity_sensors:bool=False, use_controller_building_sensors:bool=False, extra_channel_info:list=None):
    """
    Prepare predictor variable names, and instructions for feature engineering and model channel groups.
    Extra channel info should be a list of tuples in the form (channel_id, description_pattern, missing_room_ratio).
    Based on the missing_room_ratio, the function decides whether to use room-wise aggregation or not for the extra channels.

    Args:
        data_dir (str): Directory containing the metadata.parquet file.
        target_variable_name (str): Name of the target variable.
        interactive (bool): If True, prompt user for confirmation before proceeding.
        use_cooler_valves (bool): If True, include cooler valve sensors as predictors.
        use_active_setpoints (bool): If True, include active setpoint sensors as predictors.
        use_fc_room_temps (bool): If True, include FC room temperature sensors as predictors.
        use_rc_room_temps (bool): If True, include RC room temperature sensors as predictors.
        use_co2_concentrations (bool): If True, include CO2 concentration sensors as predictors.
        use_humidity_sensors (bool): If True, include humidity sensors as predictors.
        use_controller_building_sensors (bool): If True, include controller building sensors as predictors.
        extra_channel_info (list): List of tuples with extra channel info in the form (channel_id, description_pattern, missing_room_ratio).
    Returns:
        tuple: (EXAMPLE_PREDICTOR_VARIABLE_NAMES, ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES, ROOMWISE_GROUPINGS, MODEL_CHANNEL_GROUPS, extra_ids_count)

        EXAMPLE_PREDICTOR_VARIABLE_NAMES (list): List of all predictor variable names.
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES (list): List of predictor variable names that are room-wise aggregated only.
        ROOMWISE_GROUPINGS (dict): Dictionary mapping between rooms and sensor IDs for that room for each channel group.
        MODEL_CHANNEL_GROUPS (List(list)): List of final features for each channel group. For some these are the original object IDs, but for those where room-wise aggregation is used, these are the room names. 
        extra_ids_count (int): Count of extra sensor IDs added from extra_channel_info.
    """
    
    metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata = select_potential_sensors_for_test_period(metadata, missing_ratio_threshold=0.2, nunique_count_threshold=2)
    
    EXAMPLE_PREDICTOR_VARIABLE_NAMES = [
        "B205WC000.AM01",  # a supply temperature chilled water
        "B106WS01.AM54",  # an external temperature
    ]  # example predictor variables
    ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES = []
    ROOMWISE_GROUPINGS = {}
    MODEL_CHANNEL_GROUPS = []

    if use_cooler_valves:
        cooler_valves = get_cooler_valves(metadata, enable_rooms=True)
        cooler_valves_ids = cooler_valves['object_id'].unique().tolist()
        print(f"Using {len(cooler_valves_ids)} cooler valves as predictor variables.")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += cooler_valves_ids
        MODEL_CHANNEL_GROUPS.append(('cooler valves', cooler_valves_ids))

    if use_active_setpoints:
        active_setpoints = get_active_setpoints(metadata).drop_duplicates(subset=['object_id'])
        active_setpoints['room'] = active_setpoints['room'].fillna('NoRoom')
        setpoints_by_room = active_setpoints.groupby('room')['object_id'].apply(list)
        active_setpoints_ids = active_setpoints['object_id'].unique().tolist()
        print(f"Using {len(active_setpoints_ids)} active setpoints as predictor variables.")
        print(f"Number of rooms with active setpoints: {len(setpoints_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += active_setpoints_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += active_setpoints_ids
        ROOMWISE_GROUPINGS['AvgActiveSetpoint_'] = setpoints_by_room
        MODEL_CHANNEL_GROUPS.append(('active setpoints', setpoints_by_room.index.tolist()))

    if use_fc_room_temps:
        fc_room_temps = get_room_temperatures(metadata, class_id='FC').copy()
        fc_room_temps['room'] = fc_room_temps['room'].fillna('NoRoom')
        fc_room_temp_ids = fc_room_temps['object_id'].unique().tolist()
        fc_temp_by_room = fc_room_temps.groupby('room')['object_id'].apply(list)
        print(f"Using {len(fc_room_temp_ids)} FC room temperature sensors as predictor variables.")
        print(f"Number of rooms with FC room temperature sensors: {len(fc_temp_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += fc_room_temp_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += fc_room_temp_ids
        ROOMWISE_GROUPINGS['AvgFCRoomTemp_'] = fc_temp_by_room
        MODEL_CHANNEL_GROUPS.append(('FC room temperatures', fc_temp_by_room.index.tolist()))

    if use_rc_room_temps:
        rc_room_temps = get_room_temperatures(metadata, class_id='RC').copy()
        rc_room_temps['room'] = rc_room_temps['room'].fillna('NoRoom')
        rc_room_temp_ids = rc_room_temps['object_id'].unique().tolist()
        rc_temp_by_room = rc_room_temps.groupby('room')['object_id'].apply(list)
        print(f"Using {len(rc_room_temp_ids)} RC room temperature sensors as predictor variables.")
        print(f"Number of rooms with RC room temperature sensors: {len(rc_temp_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += rc_room_temp_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += rc_room_temp_ids
        ROOMWISE_GROUPINGS['AvgRCRoomTemp_'] = rc_temp_by_room
        MODEL_CHANNEL_GROUPS.append(('RC room temperatures', rc_temp_by_room.index.tolist()))

    if use_co2_concentrations:
        co2_concentration_sensors = get_co2_concentrations(metadata).copy()
        co2_concentration_sensors['room'] = co2_concentration_sensors['room'].fillna('NoRoom')
        co2_concentration_by_room = co2_concentration_sensors.groupby('room')['object_id'].apply(list)
        co2_concentration_ids = co2_concentration_sensors['object_id'].unique().tolist()
        print(f"Using {len(co2_concentration_ids)} CO2 concentration sensors as predictor variables.")
        print(f"Number of rooms with CO2 concentration sensors: {len(co2_concentration_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += co2_concentration_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += co2_concentration_ids
        ROOMWISE_GROUPINGS['AvgCO2Concentration_'] = co2_concentration_by_room
        MODEL_CHANNEL_GROUPS.append(('CO2 concentrations', co2_concentration_by_room.index.tolist()))

    if use_humidity_sensors:
        humidity_sensors = get_humidity_sensors(metadata)
        humidity_sensor_ids = humidity_sensors['object_id'].unique().tolist()
        print(f"Using {len(humidity_sensor_ids)} humidity sensors as predictor variables.")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += humidity_sensor_ids
        MODEL_CHANNEL_GROUPS.append(('humidity sensors', humidity_sensor_ids))

    if use_controller_building_sensors:
        controller_building_sensors = get_controller_building_sensors(metadata, building_id='B205')
        controller_building_sensor_ids = controller_building_sensors['object_id'].unique().tolist()
        #removing already used predictor variables
        controller_building_sensor_ids = list(set(controller_building_sensor_ids) - set(EXAMPLE_PREDICTOR_VARIABLE_NAMES))
        if target_variable_name in controller_building_sensor_ids:
            controller_building_sensor_ids.remove(target_variable_name)
        print(f"Using {len(controller_building_sensor_ids)} controller building B205 sensors as predictor variables.")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += controller_building_sensor_ids
        MODEL_CHANNEL_GROUPS.append(('controller building B205 sensors', controller_building_sensor_ids))

    extra_ids_count = 0
    if extra_channel_info:
        for channel_id, description_pattern, missing_room_ratio in extra_channel_info:
            extra_sensors = get_channel_sensors_with_description(metadata, channel_id, description_pattern).copy()
            extra_ids = extra_sensors['object_id'].unique().tolist()
            print(f"Using {len(extra_ids)} extra sensors from channel {channel_id} with description pattern '{description_pattern}' as predictor variables.")
            EXAMPLE_PREDICTOR_VARIABLE_NAMES += extra_ids
            if missing_room_ratio < 0.1:
                extra_by_rooms = extra_sensors.groupby('room')['object_id'].apply(list)
                print(f"Number of rooms with extra sensors: {len(extra_by_rooms)}")
                ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += extra_ids
                ROOMWISE_GROUPINGS[f'Avg{channel_id}_{description_pattern}_'] = extra_by_rooms
                MODEL_CHANNEL_GROUPS.append((f'extra sensors {channel_id} {description_pattern}', extra_by_rooms.index.tolist()))
            else:
                MODEL_CHANNEL_GROUPS.append((f'extra sensors {channel_id} {description_pattern}', extra_ids))
            extra_ids_count += len(extra_ids)

    print("Predictor variables:")
    #print(EXAMPLE_PREDICTOR_VARIABLE_NAMES)
    print(f"Using {len(EXAMPLE_PREDICTOR_VARIABLE_NAMES)} predictor variables.")
    if interactive:
        print('Do you want to proceed? (y/n)')
        proceed = input()
        if proceed.lower() != 'y':
            print("Exiting.")
            exit()

    return EXAMPLE_PREDICTOR_VARIABLE_NAMES, ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES, ROOMWISE_GROUPINGS, MODEL_CHANNEL_GROUPS, extra_ids_count
    