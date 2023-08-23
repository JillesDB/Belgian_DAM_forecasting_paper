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
    feature_set = ['Price']
    for num_features in range(n):
        metric_list = [] # Choose appropriate metric based on business problem
         # You can choose any model you like, this technique is model agnostic
        for feature in dataframe.columns[1:]:
            if feature not in feature_set:
                feature_set.append(feature)
                ffs_dataframe = dataframe[feature_set].copy()
                ffs_dataframe.to_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\ffs_dataframe.csv')
                Forecasting.create_single_forecast(name_dataframe='ffs_dataframe',calibration_window=calibration_window,begin_test_date=begin_test_date,end_test_date=end_test_date)
                mae_model = Evaluate_forecast.calc_mae(file_forecast='LEAR_forecast' + '_dataframe_' + str(ffs_dataframe) + '_CW56_RW1.csv',
                                                      path_real_prices='C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv')
                metric_list.append((mae_model, feature))

        metric_list.sort(key=lambda x : x[0])#, reverse = True) # In case metric follows "the more, the merrier"
        feature_set.append(metric_list[0][1])
    return feature_set

f = forward_feature_selection(path_dataframe=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Example_dataframe.csv',
                              begin_test_date='2020-01-01',end_test_date='2020-03-31',calibration_window=56,n=2)
print(f)