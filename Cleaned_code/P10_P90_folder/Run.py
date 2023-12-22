
import os

import pandas as pd

import Generate_Evaluate_and_Time_Forecast_P10_P90
from pathlib import Path

cwd = Path.cwd()
cleaned_code_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(cwd, 'Datasets\P10_P90_datasets')
forecasts_folder = os.path.join(cwd, 'Forecasts')
real_prices_path = os.path.join(cwd, 'Datasets\Real_prices.csv')


# df = pd.read_csv(r'C:\Users\jdeblauw\Documents\GitHub\Belgian_DAM_forecasting_paper\Cleaned_code\P10_P90_folder\Forecasts2\LEAR_train_dataframe_Datasettest_dataframe_Dataset_CW714_RW1.csv')
# df2 = pd.read_csv(r'C:\Users\jdeblauw\Documents\GitHub\Belgian_DAM_forecasting_paper\Cleaned_code\P10_P90_folder\Forecasts2\LEAR_train_dataframe_Datasettest_dataframe_Dataset_P10_Windsolar_P90_Load_CW714_RW1.csv')
# print(df.mean(),df2.mean())

Generate_Evaluate_and_Time_Forecast_P10_P90.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder = os.path.join(cwd, 'Forecasts')
,dataset_train='Dataset',dataset_test='Dataset_P10_Windsolar_Reg_Load',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2020-02-01',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)

Generate_Evaluate_and_Time_Forecast_P10_P90.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,dataset_train='Dataset',dataset_test='Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)

Generate_Evaluate_and_Time_Forecast_P10_P90.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,dataset_train='Dataset',dataset_test='Dataset_P90_Windsolar_P10_Load',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)



