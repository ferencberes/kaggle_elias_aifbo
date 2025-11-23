import pandas as pd
import os

def extract_channel_group_information(data_dir, output_fp, min_sensor_count=20):
    metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata['description_without_room'] = metadata.apply(lambda row: row['description'].replace(str(row['room']), '') if pd.notnull(row['room']) else row['description'], axis=1)
    metadata['short_description'] = metadata['description_without_room'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    channel_counts = metadata['channel'].value_counts()
    selected = channel_counts[channel_counts > min_sensor_count]
    most_pop_short_desc_info = {}
    missing_room_ratio = {}
    for channel_id in selected.index:
        top_short_counts = metadata[metadata['channel'] == channel_id]['short_description'].value_counts()
        top_short_counts = top_short_counts / top_short_counts.sum()
        most_pop_short_desc_info[channel_id] = top_short_counts.index[0] + f" ({top_short_counts.iloc[0]*100:.1f}%)"
        missing_room_ratio[channel_id] = metadata[metadata['channel'] == channel_id]['room'].isna().mean()
    channel_info_df = pd.DataFrame({'most_popular_short_description': most_pop_short_desc_info, 'missing_room_ratio': missing_room_ratio})
    selected = pd.DataFrame(selected).join(channel_info_df)
    selected.columns = ['sensor_count', 'most_popular_short_description', 'missing_room_ratio']
    print(selected.head())
    selected.to_csv(output_fp, index=True)