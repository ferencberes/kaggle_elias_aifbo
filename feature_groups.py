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

# these status step ventillation variables are very scarce 
#B201FC063_1.VS01_1
#B201FC268_2.VS01_1
#B201FC368_1.VS01_1
#B201FC112_1.VS01_1
#B201FC475_1.VS01_1
#B201FC223_1.VS01_1

import pandas as pd

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