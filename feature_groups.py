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

import pandas as pd
import os

def aggregate_sensor_values(selected_sensor_metadata, timeseries_df, key_col, id_col='object_id', value_agg_func='mean', weight_col=None, weight_agg_func='max'):
    sensors_by_key = selected_sensor_metadata.groupby(key_col)[id_col].apply(list).to_dict()
    if weight_col:
        weight_by_key = selected_sensor_metadata.groupby(key_col)[weight_col].agg(weight_agg_func).to_dict()
        sum_weight = sum(weight_by_key.values())
    columns = set(timeseries_df.columns)
    new_feature = pd.Series(0, index=timeseries_df.index, dtype=float)
    for key, sensors in sensors_by_key.items():
        available_sensors = list(set(sensors).intersection(columns))
        if not available_sensors:
            continue
        agg_value = timeseries_df[available_sensors].agg(value_agg_func, axis=1)
        if weight_col:
            weight = weight_by_key[key] / sum_weight
        else:
            weight = 1.0 / len(sensors_by_key)
        new_feature += agg_value * weight
    return new_feature

def add_room_return_temp_features(metadata, timeseries_df):
    return_temp_features = metadata[(metadata['building']=='B201') & (metadata['bde_group']=='RC') & (metadata['dimension_index'] == 2.0) & (metadata['conversion_index'] == 118.0)]
    return_room_temp = return_temp_features[return_temp_features['description'].str.upper().str.startswith('ROOM TEMPERATURE')].copy()
    
    if return_room_temp.empty:
        print("No return room temperature sensors found for B201 RC")
        return timeseries_df
    
    return_room_temp['bim_room_description'] = return_room_temp['bim_room_description'].fillna('NoDescription')
    return_room_temp['bim_room_category'] = return_room_temp['bim_room_category'].fillna('NoCategory')
    return_room_temp['bim_energy_category'] = return_room_temp['bim_energy_category'].fillna('NoEnergyCategory')

    temp_by_room_type = {}
    for col in ['bim_room_description', 'bim_energy_category']:#, 'bim_room_category']:
        print(f"{col}: {return_room_temp[col].nunique()} unique values")
        room_meta = return_room_temp[col].unique()
        for description in room_meta:
            desc_metadata = return_room_temp[return_room_temp[col] == description]
            weighted_temp = aggregate_sensor_values(desc_metadata, timeseries_df, key_col='room', weight_col='bim_room_area')# avg room temp weighted by room area
            temp_by_room_type[description] = weighted_temp

    temp_by_room_type['AllRooms'] = aggregate_sensor_values(return_room_temp, timeseries_df, key_col='room', weight_col='bim_room_area')# avg room temp weighted by room area

    new_feature_cols = []
    for description, temp_series in temp_by_room_type.items():
        feature_name = f'B201_RC_ReturnRoomTemp_{description.replace(" ", "_").replace("/", "_")}'
        timeseries_df[feature_name] = temp_series
        new_feature_cols.append(feature_name)
        #print(f"Added feature: {feature_name}")

    return timeseries_df, new_feature_cols

def get_cooler_valves(metadata, enable_rooms=False):
    cooler_valves = metadata[metadata['channel']=='AC21']
    if enable_rooms:
        return cooler_valves
    else:
        return cooler_valves[cooler_valves['room'].isnull()]
    
def get_active_setpoints(metadata, k=None):
    # room information is always available for VT03_2 setpoints
    active_setpoints = metadata[metadata['channel']=='VT03_2']
    if k is not None:
        active_setpoints = active_setpoints.nlargest(k, 'bim_room_area')
    return active_setpoints

def get_co2_concentrations(metadata, k=None):
    co2_sensors = metadata[(metadata['channel']=='AM21') & (metadata['dimension_text'].str.lower()=='ppm')]
    #sometimes CO2 concentration has AM22 channels as well.. so I have more sensors but no extra room attributed to this change
    #co2_sensors = metadata[(metadata['channel'].isin(['AM21', 'AM22'])) & (metadata['dimension_text'].str.lower()=='ppm')]
    # it made results worse!!!
    if k is not None:
        co2_sensors = co2_sensors.nlargest(k, 'bim_room_area')
    return co2_sensors

def get_humidity_sensors(metadata, k=None):
    humidity_sensors = metadata[metadata['channel'].isin(['AM45', 'AM45_1', 'AM51'])]
    excluded_ids = [
        'B106WS01.AM51', #light intensity sensor
    ]
    humidity_sensors = humidity_sensors[humidity_sensors['object_id'].isin(excluded_ids) == False]
    if k is not None:
        humidity_sensors = humidity_sensors.nlargest(k, 'bim_room_area')
    return humidity_sensors

def get_controller_building_sensors(metadata, building_id='B205', excluded_channels=['AC21', 'VT03_2', 'AM21', 'AM45', 'AM45_1', 'AM51']):
    building_sensors = metadata[metadata['object_id'].str.startswith(building_id)] 
    if excluded_channels:
        building_sensors = building_sensors[~building_sensors['channel'].isin(excluded_channels)]
    return building_sensors

def get_room_temperatures(metadata, class_id='FC', k=None):
    fancoil_temps = metadata[(metadata['object_id'].str.startswith('B201'+class_id)) & (metadata['description'].str.upper().str.contains('ROOM TEMPERATURE')) & (metadata['channel']=='AM01')]
    if k is not None:
        fancoil_temps = fancoil_temps.nlargest(k, 'bim_room_area')
    return fancoil_temps

def prepare_predictor_variables(data_dir, TARGET_VARIABLE_NAME, interactive=True, use_cooler_valves=True, use_active_setpoints=False, use_fc_room_temps=False, use_rc_room_temps=False, use_co2_concentrations=False, use_humidity_sensors=False, use_controller_building_sensors=False):
    
    metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    feature_information = pd.read_csv('test_set_feature_information.csv', index_col=0)
    potential_features = feature_information[
        (feature_information['missing_ratio'] < 0.2) &
        (feature_information['nunique_count'] > 2)
    ].copy()
    potential_features = potential_features.index.tolist()
    # keep only features that are available during the test set period
    metadata = metadata[metadata['object_id'].isin(potential_features)].copy()

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
        MODEL_CHANNEL_GROUPS.append(('active setpoints', active_setpoints_ids))

    if use_fc_room_temps:
        fc_room_temps = get_room_temperatures(metadata, class_id='FC')
        fc_room_temps['room'] = fc_room_temps['room'].fillna('NoRoom')
        fc_room_temp_ids = fc_room_temps['object_id'].unique().tolist()
        fc_temp_by_room = fc_room_temps.groupby('room')['object_id'].apply(list)
        print(f"Using {len(fc_room_temp_ids)} FC room temperature sensors as predictor variables.")
        print(f"Number of rooms with FC room temperature sensors: {len(fc_temp_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += fc_room_temp_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += fc_room_temp_ids
        ROOMWISE_GROUPINGS['AvgFCRoomTemp_'] = fc_temp_by_room
        MODEL_CHANNEL_GROUPS.append(('FC room temperatures', fc_room_temp_ids))

    if use_rc_room_temps:
        rc_room_temps = get_room_temperatures(metadata, class_id='RC')
        rc_room_temps['room'] = rc_room_temps['room'].fillna('NoRoom')
        rc_room_temp_ids = rc_room_temps['object_id'].unique().tolist()
        rc_temp_by_room = rc_room_temps.groupby('room')['object_id'].apply(list)
        print(f"Using {len(rc_room_temp_ids)} RC room temperature sensors as predictor variables.")
        print(f"Number of rooms with RC room temperature sensors: {len(rc_temp_by_room)}")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += rc_room_temp_ids
        ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES += rc_room_temp_ids
        ROOMWISE_GROUPINGS['AvgRCRoomTemp_'] = rc_temp_by_room
        MODEL_CHANNEL_GROUPS.append(('RC room temperatures', rc_room_temp_ids))

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
        MODEL_CHANNEL_GROUPS.append(('CO2 concentrations', co2_concentration_ids))

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
        if TARGET_VARIABLE_NAME in controller_building_sensor_ids:
            controller_building_sensor_ids.remove(TARGET_VARIABLE_NAME)
        print(f"Using {len(controller_building_sensor_ids)} controller building B205 sensors as predictor variables.")
        EXAMPLE_PREDICTOR_VARIABLE_NAMES += controller_building_sensor_ids
        MODEL_CHANNEL_GROUPS.append(('controller building B205 sensors', controller_building_sensor_ids))

    print("Predictor variables:")
    #print(EXAMPLE_PREDICTOR_VARIABLE_NAMES)
    print(f"Using {len(EXAMPLE_PREDICTOR_VARIABLE_NAMES)} predictor variables.")
    if interactive:
        print('Do you want to proceed? (y/n)')
        proceed = input()
        if proceed.lower() != 'y':
            print("Exiting.")
            exit()

    return EXAMPLE_PREDICTOR_VARIABLE_NAMES, ROOMWISE_ONLY_PREDICTOR_VARIABLE_NAMES, ROOMWISE_GROUPINGS, MODEL_CHANNEL_GROUPS
    