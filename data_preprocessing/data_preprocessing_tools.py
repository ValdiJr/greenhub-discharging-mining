# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import datetime as dt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../csv_folder/csv_top_device_merged.csv')
dataset["consume_energy_timestamp"] = dataset["consume_energy_timestamp"].apply(pd.to_timedelta, unit='s')
dataset["consume_energy_timestamp"] = dataset["consume_energy_timestamp"].dt.total_seconds()
dataset["consume_energy_timestamp"] = dataset["consume_energy_timestamp"].loc[(dataset["consume_energy_timestamp"] > 0)]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Unn')))]
dataset=dataset[(dataset["first_battery_level"]-dataset["last_battery_level"])<0.02]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='_id')))]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='last_')))]
target = dataset["consume_energy_timestamp"]

dataset["network_type"].value_counts().tail(1).index[0]
categorical = ['network_type', 'mobile_network_type','charger','health',
       'mobile_data_status', 'roaming_enabled', 'wifi_status',
       'wifi_ap_status', 'network_operator', 'sim_operator', 'provider']

for col in categorical:
    feature = dataset[col].value_counts().tail(1).index[0]
    dataset = pd.get_dummies(dataset, columns=[col])
    dataset = dataset.drop([col+"_"+str(feature)], axis=1)
    
numerical = ['bluetooth_enabled', 'location_enabled',
       'power_saver_enabled', 'nfc_enabled', 'flashlight_enabled']

for col in numerical:
    feature = dataset[col].value_counts().tail(1).index[0]
    dataset = pd.get_dummies(dataset, columns=[col])
    dataset = dataset.drop([col+"_"+str(feature)], axis=1)
    

dataset=pd.concat([dataset,target], axis=1)
dataset.to_csv('../csv_folder/csv_top_device_merged.csv')
