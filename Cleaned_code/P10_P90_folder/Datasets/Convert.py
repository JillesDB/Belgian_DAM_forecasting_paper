import datetime

from Epftoolbox_original_code import _lear
import pandas as pd
import os
import numpy as np
from pathlib import Path

# df = pd.read_csv(os.path.join(Path.cwd(), 'wind_aggregated.csv'),delimiter=',')
# print(df)
# df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
# df = df.set_index('Datetime')
# df = df.sort_index()
# print(df)
# df.to_csv(os.path.join(Path.cwd(),'wind_aggregated.csv'))
df = pd.read_csv(os.path.join(Path.cwd(), 'Datasets_Zeynep/converted/solar_belgium.csv'), delimiter=',')
df2 = pd.read_csv(os.path.join(Path.cwd(), 'Datasets_Zeynep/converted/wind_aggregated.csv'), delimiter=',')
df3 = pd.read_csv(os.path.join(Path.cwd(), 'Datasets_Zeynep/converted/load.csv'), delimiter=',')
Full_dataset = pd.read_csv(r'C:\Users\jdeblauw\Documents\GitHub\Belgian_DAM_forecasting_paper\Cleaned_code\Datasets\Full_Dataset.csv')

#dataset no ch prices
Dataset = Full_dataset.copy(deep = False)
Dataset = Dataset.drop(columns=['CH Price'],axis =1)
# Dataset = Dataset.set_index('Date')
print(Dataset.iloc[36],df2['Day Ahead 11AM P10'][36],df['Day Ahead 11AM P10'],Dataset)

#dataset P10 renewables, regular load
Dataset_P10_Windsolar_Reg_Load = Dataset.copy(deep = False)
Dataset_P10_Windsolar_Reg_Load.insert(3,'BE DA P10 Wind',value = df2['Day Ahead 11AM P10'])
print(Dataset_P10_Windsolar_Reg_Load.iloc[36])
Dataset_P10_Windsolar_Reg_Load.insert(4,'BE DA P10 Solar',value = df['Day Ahead 11AM P10'])
Dataset_P10_Windsolar_Reg_Load = Dataset_P10_Windsolar_Reg_Load.drop(columns=['BE Wind','BE Solar'],axis =1)
print(Dataset_P10_Windsolar_Reg_Load.iloc[36])
Dataset_P10_Windsolar_Reg_Load = Dataset_P10_Windsolar_Reg_Load.fillna(method = 'ffill')

#dataset P10 renewables, p90 load
Dataset_P10_Windsolar_P90_Load = Dataset.copy(deep = False)
Dataset_P10_Windsolar_P90_Load.insert(3,'BE DA P10 Wind',value = df2['Day Ahead 11AM P10'])
Dataset_P10_Windsolar_P90_Load.insert(4,'BE DA P10 Solar',value = df['Day Ahead 11AM P10'])
Dataset_P10_Windsolar_P90_Load.insert(2,'BE DA P90 Load',value = df3['Day-ahead 6PM P90'])
Dataset_P10_Windsolar_P90_Load = Dataset_P10_Windsolar_P90_Load.drop(columns=['BE Wind','BE Solar','BE Load'],axis =1)
Dataset_P10_Windsolar_P90_Load = Dataset_P10_Windsolar_P90_Load.fillna(method = 'ffill')


#dataset P90 renewables, regular load
Dataset_P90_Windsolar_Reg_Load = Dataset.copy(deep = False)
Dataset_P90_Windsolar_Reg_Load.insert(3,'BE DA P90 Wind',value = df2['Day Ahead 11AM P90'])
Dataset_P90_Windsolar_Reg_Load.insert(4,'BE DA P90 Solar',value = df['Day Ahead 11AM P90'])
Dataset_P90_Windsolar_Reg_Load = Dataset_P90_Windsolar_Reg_Load.drop(columns=['BE Wind','BE Solar'],axis =1)
Dataset_P90_Windsolar_Reg_Load = Dataset_P90_Windsolar_Reg_Load.fillna(method = 'ffill')

#dataset P90 renewables, p10 load
Dataset_P90_Windsolar_P10_Load = Dataset.copy(deep = False)
Dataset_P90_Windsolar_P10_Load.insert(3,'BE DA P90 Wind',value = df2['Day Ahead 11AM P90'])
Dataset_P90_Windsolar_P10_Load.insert(4,'BE DA P90 Solar',value = df['Day Ahead 11AM P90'])
Dataset_P90_Windsolar_P10_Load.insert(2,'BE DA P10 Load',value = df3['Day-ahead 6PM P10'])
Dataset_P90_Windsolar_P10_Load = Dataset_P90_Windsolar_P10_Load.drop(columns=['BE Wind','BE Solar','BE Load'],axis =1)
Dataset_P90_Windsolar_P10_Load = Dataset_P90_Windsolar_P10_Load.fillna(method = 'ffill')

Dataset = Dataset.set_index('Date')
Dataset_P10_Windsolar_Reg_Load = Dataset_P10_Windsolar_Reg_Load.set_index('Date')
Dataset_P90_Windsolar_P10_Load = Dataset_P90_Windsolar_P10_Load.set_index('Date')
Dataset_P10_Windsolar_P90_Load = Dataset_P10_Windsolar_P90_Load.set_index('Date')
Dataset_P90_Windsolar_Reg_Load = Dataset_P90_Windsolar_Reg_Load.set_index('Date')

Dataset = Dataset.fillna(method  = 'ffill')
Dataset_P10_Windsolar_Reg_Load = Dataset_P10_Windsolar_Reg_Load.fillna(method  = 'ffill')
Dataset_P90_Windsolar_P10_Load = Dataset_P90_Windsolar_P10_Load.fillna(method  = 'ffill')
Dataset_P10_Windsolar_P90_Load = Dataset_P10_Windsolar_P90_Load.fillna(method  = 'ffill')
Dataset_P90_Windsolar_Reg_Load = Dataset_P90_Windsolar_Reg_Load.fillna(method  = 'ffill')
print(Dataset_P90_Windsolar_P10_Load.iloc[36],Dataset.iloc[36],Dataset_P10_Windsolar_P90_Load.iloc[36],Dataset_P90_Windsolar_Reg_Load.iloc[36],Dataset_P10_Windsolar_Reg_Load.iloc[36])

Dataset.to_csv(os.path.join(Path.cwd(), 'P10_P90_datasets\Dataset.csv'))
Dataset_P10_Windsolar_Reg_Load.to_csv(os.path.join(Path.cwd(), 'P10_P90_datasets\Dataset_P10_Windsolar_Reg_Load.csv'))
Dataset_P90_Windsolar_P10_Load.to_csv(os.path.join(Path.cwd(), 'P10_P90_datasets\Dataset_P90_Windsolar_P10_Load.csv'))
Dataset_P10_Windsolar_P90_Load.to_csv(os.path.join(Path.cwd(), 'P10_P90_datasets\Dataset_P10_Windsolar_P90_Load.csv'))
Dataset_P90_Windsolar_Reg_Load.to_csv(os.path.join(Path.cwd(), 'P10_P90_datasets\Dataset_P90_Windsolar_Reg_Load.csv'))

# Full_dataset['DA P10 Solar'],Full_dataset['DA P10 Wind'] = df['Day Ahead 11AM P10'],df2['Day Ahead 11AM P10']
# df = df.drop(df.index[0:23528])
# df = df.drop(columns='Datetime')
# df = df.groupby(np.arange(len(df))//4).mean()
# df['Date'] = df2['Date']
# # df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
# # df = df.set_index('Datetime')
# # df = df.sort_index()
# # print(df)
# df.to_csv(os.path.join(Path.cwd(),'converted\solar_belgium.csv'))

# df['Datetime'] = df['Datetime'].map(lambda x: x.rstrip('+01:00'))
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# df = pd.read_parquet(os.path.join(Path.cwd(), 'parquets/solar_belgium.parquet'))
# df.to_csv(os.path.join(Path.cwd(),'solar_belgium.csv'))
#
# df = pd.read_parquet(os.path.join(Path.cwd(), 'parquets/wind.parquet'))
# df.to_csv(os.path.join(Path.cwd(),'wind.csv'))
#
# df = pd.read_parquet(os.path.join(Path.cwd(), 'parquets/wind_aggregated.parquet'))
# df.to_csv(os.path.join(Path.cwd(),'wind_aggregated.csv'))