from Epftoolbox_original_code import _lear
import pandas as pd
import os
import numpy as np
from Epftoolbox_original_code.evaluation import _mae,_rmae
import Evaluate_forecast,Forecasting,Timing_Forecasts
from pathlib import Path


cwd = Path.cwd()
path_datasets_folder = os.path.join(cwd,'Datasets')
path_real_prices = os.path.join(path_datasets_folder,'Real_prices.csv')
path_forecasts_folder = os.path.join(cwd,'Forecasts')

def forward_feature_selection(name_dataframe,begin_test_date,end_test_date, n,calibration_window=56):
    """

def forward_feature_selection(name_dataframe,begin_test_date,end_test_date, n,calibration_window=56):
    """
    Parameters
    ----------
    name_dataframe
    begin_test_date
    end_test_date
    n
    calibration_window

    Returns
    -------

    """

    path_dataframe = os.path.join(path_datasets_folder,name_dataframe)
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
                ffs_dataframe.to_csv(os.path.join(path_datasets_folder,'ffs_dataframe.csv'))
                path_forecasts_folder = os.path.join(cwd,'Forecasts_for_plots')

                name_file = str(('LEAR_forecast_dataframe_ffs_dataframe_CW'+str(calibration_window)+'_RW1.csv'))
                Forecasting.create_single_forecast(name_dataframe='ffs_dataframe',path_forecast_folder=path_forecasts_folder,
                                                   calibration_window=calibration_window,begin_test_date=begin_test_date,end_test_date=end_test_date)
                mae_model = Evaluate_forecast.calc_mae(path_forecast=os.path.join(str(path_forecasts_folder),name_file),path_real_prices=path_real_prices)
                metric_list.append((mae_model, feature_family))
        metric_list.sort(key=lambda x : x[0])#, reverse = True) # In case metric follows "the more, the merrier"
        print(metric_list)
        feature_set.extend(metric_list[0][1])
    return feature_set

f = forward_feature_selection(name_dataframe='Full_Dataset.csv',begin_test_date='2020-01-01',end_test_date='2022-12-31',calibration_window=728,n=13)
print(f)

