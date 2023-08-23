#THIS FILE IS WORK IN PROGRESS
from Epftoolbox_original_code import _lear
import pandas as pd
import os
import numpy as np
from Epftoolbox_original_code.evaluation import _mae,_rmae
import Evaluate_forecast,Forecasting,Timing_Forecasts
from pathlib import Path

cwd = Path.cwd()
path_forecasts_folder = str(cwd) + '\Forecasts'
def Predict_Evaluate_and_Time(path_real_prices,name_dataframe,path_datasets_folder,
                              begin_test_date,end_test_date,
                              recalibration_window = 1,calibration_window_set=frozenset([56,84,112,714,721,728]),
                              weighed = 0 ,regular = 0):
    """

    Parameters
    ----------
    path_real_prices
    name_dataframe
    begin_test_date
    end_test_date
    calibration_window

    Returns
    -------

    """
    # dataframe = pd.read_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Example_dataframe.csv')
    # real_prices = pd.read_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv')
    # dataframe = dataframe.set_index('Date')
    # real_prices = real_prices.set_index('Date')
    Dataframe_evaluation_and_timing = pd.DataFrame(0,index=list('CW ' + str(cw) for cw in calibration_window_set),columns=['MAE','rMAE','Time'])
    dict_timing = Forecasting.create_ensemble_forecast(name_dataframe=name_dataframe,path_real_prices=path_real_prices,path_datasets_folder=path_datasets_folder,
                                         begin_test_date=begin_test_date,end_test_date=end_test_date,recalibration_window=1,weighed=weighed,regular=regular,return_time=1)

    for cw in calibration_window_set:
        name_csv_file = 'LEAR_forecast' + '_dataframe_' + str(name_dataframe) + \
                         '_CW' + str(cw) + '_RW' + str(recalibration_window) + '.csv'
        path_file = os.path.join(path_forecasts_folder,name_csv_file)
        Dataframe_evaluation_and_timing.loc['CW ' + str(cw),'MAE'] = Evaluate_forecast.calc_mae(path_file,path_real_prices=path_real_prices)
        Dataframe_evaluation_and_timing.loc['CW ' + str(cw), 'rMAE'] = Evaluate_forecast.calc_rmae(path_file,path_real_prices=path_real_prices)
        if len(dict_timing) !=1:
            print(dict_timing)
            Dataframe_evaluation_and_timing.loc['CW ' + str(cw),'Time'] = dict_timing['Time CW ' + str(cw)]

    print('Results for dataframe {}  from {} until {}, RW {}'.format(name_dataframe,begin_test_date, end_test_date,recalibration_window))
    print(Dataframe_evaluation_and_timing)
