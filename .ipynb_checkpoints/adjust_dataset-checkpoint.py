import pandas as pd
import numpy as np
import datetime
import math

import reshape_ev_charging

def calc_price_peak_off_peak(time_of_day, day_of_week):
    peak_price = 0.08
    mid_peak_price = 0.075
    off_peak_price = 0.07
    if day_of_week in (6,7) or 0 < time_of_day * 24 < 8:
        return off_peak_price
    elif 10 < time_of_day * 24 < 14 or 18 < time_of_day * 24 < 22:
        return peak_price
    return mid_peak_price



def str_to_timestamp(s):
    y = int(s[0:4])
    m = int(s[5:7])
    d = int(s[8:10])
    h = int(s[11:13])
    minute = int(s[14:16])
    minute = minute - (minute % 15)
    sec = int(s[17:19])
    
    # no seconds, because we will be joining columns on date
    dt = datetime.datetime(y, m, d, h, minute)
    
    return dt
    
def reformat_table(ts, df):
    timestamps = []
    last_smp_value = None
    for ix, row in df.iterrows():
        
        timestamps.append(str_to_timestamp(row['Timestamp_UTC']))
    
    df.insert(0, 'Timestamp', timestamps)
    df = df.drop(columns=['Timestamp_UTC'])
    
      
    df = df.groupby(['Timestamp']).mean()
    df = ts.merge(df, how='left', on='Timestamp')
    
    if 'SMP' in df.columns:
        for ix, row in df.iterrows():
            if row['Timestamp'].minute == 0:
                last_smp_value = row['SMP']
            else:
                df.at[ix, 'SMP'] = last_smp_value
    
    return df

def get_time_15min():
    # got these values from the dataset
    initial_timestamp = datetime.datetime(2018, 10, 3, 21, 0)
    final_timestamp = datetime.datetime(2020, 9, 30, 20, 45)
    timestamp_increment = initial_timestamp
    timestamps = []
    while timestamp_increment < final_timestamp:
        timestamps.append(timestamp_increment)
        timestamp_increment += datetime.timedelta(minutes=15)
        
    return pd.DataFrame(timestamps).rename(columns={0: 'Timestamp'})


ts = get_time_15min()

battery_energy = reformat_table(ts, pd.read_csv('data/Battery_Energy_Measurements_03.10.2018_30.09.2020.csv'))
load_consumption = reformat_table(ts, pd.read_csv('data/Load_Consumption_Measurements_03.10.2018_30.09.2020.csv'))
pv_generation = reformat_table(ts, pd.read_csv('data/PV_Generation_Measurements_03.10.2018_30.09.2020.csv'))
smp = reformat_table(ts, pd.read_csv('data/SMP_Measurements_03.10.2018_30.09.2020.csv'))
weather = reformat_table(ts, pd.read_csv('data/Weather_Measurements_03.10.2018_30.09.2020.csv'))
tables = [battery_energy, load_consumption, pv_generation, smp, weather]

ev_charging = pd.read_csv('data/ev_charging_reshaped.csv')


# join on timestamp
df = ts.merge(battery_energy, how='left', on='Timestamp')
df = df.merge(load_consumption, how='left', on='Timestamp')
df = df.merge(pv_generation, how='left', on='Timestamp')
df = df.merge(smp, how='left', on='Timestamp')
df = df.merge(weather, how='left', on='Timestamp')
df = df.sort_values('Timestamp')

ev = reshape_ev_charging.get_table()
df = df.merge(ev, how='left', on='Timestamp')
print(df)


# drop nan rows and select relevant columns
df = df.dropna(subset=['Energy_Generation', 'Energy_Consumption', 'SMP', 'EV_Consumption'])
df = df[['Timestamp', 'Energy_Generation', 'Energy_Consumption', 'SMP', 'EV_Consumption']]
df = df.loc[(df['SMP'] < 1) & (df['Energy_Consumption'] < 10)]



maxs = df.max().to_frame().rename(columns={0: 'max'})
mins = df.min().to_frame().rename(columns={0: 'min'})
nans = df.isna().sum().to_frame().rename(columns={0: 'nan count'})
df_info = maxs.join(mins).join(nans)

t = list(df['Timestamp'])
time_of_day = []
day_of_week = []
for timestamp in t:
    time_of_day.append(float((timestamp.hour * 4 + timestamp.minute / 15) / (24 * 4)))
    day_of_week.append(float(timestamp.day))

df['Time_of_Day'] = time_of_day
df['Day_of_Week'] = day_of_week

'''
smp = np.array(df['SMP'])
minn = min(smp)
print(minn)
mean = smp.mean()
print(mean)
var_smp = smp.var()
print(smp)
#smp = ((smp - mean) * 10) + mean
smp = smp**2
print(smp)
'''


time = np.array([df['Time_of_Day'], df['Day_of_Week']]).T
#smp = np.array([calc_price_peak_off_peak(x[0], x[1]) for x in time])
#df['SMP'] = smp


df['Energy_Generation'] = [0]*len(df)



df.to_csv('working_data_no_gen.csv')
