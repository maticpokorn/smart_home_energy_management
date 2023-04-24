import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_time_15min(y):
    initial_timestamp = datetime.datetime(y, 1, 1, 0, 0)
    final_timestamp = datetime.datetime(y + 1, 1, 1, 0, 0)
    timestamp_increment = initial_timestamp
    timestamps = []
    while timestamp_increment < final_timestamp:
        timestamps.append(pd.Timestamp(timestamp_increment))
        timestamp_increment += datetime.timedelta(minutes=15)

    return pd.DataFrame(timestamps, columns=['Timestamp'])


def get_table_y(y):
    # this csv file stores ev charging power data in a 30min interval. In order for it to be compatible with the other
    # dataset with 15min intervals (Greek house), we artificially insert another timestamp as the mean between two data
    # points, which already exist
    df = pd.read_csv('data/ev_charging.csv', header=None)
    rows = []
    print("initial length:", len(df))
    for i in tqdm(range(len(df) - 1)):
        row = np.array(df.loc[i, :])
        row_next = np.array(df.loc[i + 1, :])
        row_middle = (row + row_next) / 2
        rows.append(row)
        rows.append(row_middle)

    last_row = np.array(df.loc[len(df) - 1, :])
    rows.append(last_row)
    rows.append(last_row)

    print("desired length:", len(df) * 2 - 1)
    print("real length:", np.array(rows).shape[0])
    print(len(get_time_15min(2018)))

    my_df = pd.DataFrame(np.array(rows))
    my_df['EV_Consumption'] = my_df[1] / 4
    my_df = pd.concat([get_time_15min(y), my_df], axis=1)
    print(my_df.dtypes)
    #my_df.to_csv('data/ev_charging_reshaped.csv', index=False)
    return my_df


def get_table():
    my_df = pd.concat([get_table_y(2018), get_table_y(2019), get_table_y(2020)])
    my_df.to_csv('data/ev_charging_reshaped.csv', index=False)
    return my_df
