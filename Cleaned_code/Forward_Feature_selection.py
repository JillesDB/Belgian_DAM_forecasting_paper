from Epftoolbox_original_code import _lear
import pandas as pd
import os
import numpy as np
from Epftoolbox_original_code.evaluation import _mae,_rmae
import Evaluate_forecast,Forecasting,Timing_Forecasts
from pathlib import Path




def forward_feature_selection(path_dataframe,begin_test_date,end_test_date, n,calibration_window=56):
    dataframe = pd.read_csv(path_dataframe)
    dataframe = dataframe.set_index('Date')
    feature_set = ['Price','CH Price']

    while len(feature_set) < n:
        metric_list = []  # Choose appropriate metric based on business problem
        for feature_family in (['FR Generation','FR Load'],['BE Load','Av. Hourly Temp'],['BE Solar','DE Solar'],
                              ['BE Wind','DE Wind'],['Brent Oil','Carbon Price','TTF NG Price']):
            # You can choose any model you like, this technique is model agnostic
            if feature_family[0] not in set(feature_set):
                test_set = feature_set.copy()
                test_set.extend(feature_family)
                print(test_set)
                ffs_dataframe = dataframe[test_set].copy()
                ffs_dataframe.to_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\ffs_dataframe.csv')
                Forecasting.create_single_forecast(name_dataframe='ffs_dataframe',path_forecast_folder=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Working_dir',
                                                   calibration_window=calibration_window,begin_test_date=begin_test_date,end_test_date=end_test_date)
                mae_model = Evaluate_forecast.calc_mae(path_forecast=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Working_dir\LEAR_forecast_dataframe_ffs_dataframe_CW112_RW1.csv',
                                                      path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv')
                metric_list.append((mae_model, feature_family))
        metric_list.sort(key=lambda x : x[0])#, reverse = True) # In case metric follows "the more, the merrier"
        print(metric_list)
        feature_set.extend(metric_list[0][1])
    return feature_set

f = forward_feature_selection(path_dataframe=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Example_dataframe.csv',
                              begin_test_date='2020-01-01',end_test_date='2022-12-31',calibration_window=112,n=13)
print(f)