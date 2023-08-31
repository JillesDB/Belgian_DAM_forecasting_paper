#THIS FILE IS WORK IN PROGRESS
from Epftoolbox_original_code import _lear
import pandas as pd
import os
import numpy as np
from Epftoolbox_original_code.evaluation import _mae,_rmae
import Evaluate_forecast,Forecasting,Timing_Forecasts
from pathlib import Path

cwd = Path.cwd()

def Predict_Evaluate_and_Time(path_real_prices,name_dataframe,path_datasets_folder,
                              begin_test_date,end_test_date,path_forecasts_folder=None,
                              recalibration_window = 1,calibration_window_set=tuple([56,84,112,714,721,728]),
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
    if path_forecasts_folder is None:
        path_forecasts_folder = str(cwd) + '\Forecasts'
    Dataframe_evaluation_and_timing = pd.DataFrame(0,index=list('CW ' + str(cw) for cw in calibration_window_set),columns=['MAE','rMAE','Time'])
    dict_timing = Forecasting.create_ensemble_forecast(name_dataframe=name_dataframe,path_real_prices=path_real_prices,path_datasets_folder=path_datasets_folder,
                                         begin_test_date=begin_test_date,end_test_date=end_test_date,path_forecasts_folder=path_forecasts_folder,
                                                       recalibration_window=recalibration_window,weighed=weighed,regular=regular,return_time=1,calibration_window_set=calibration_window_set)

    for cw in calibration_window_set:
        name_csv_file = 'LEAR_forecast' + '_dataframe_' + str(name_dataframe) + \
                         '_CW' + str(cw) + '_RW' + str(recalibration_window) + '.csv'
        path_file = os.path.join(path_forecasts_folder,name_csv_file)
        #print(Dataframe_evaluation_and_timing,path_file)
        Dataframe_evaluation_and_timing.loc['CW ' + str(cw),'MAE'] = Evaluate_forecast.calc_mae(path_forecast=path_file,path_real_prices=path_real_prices)
        Dataframe_evaluation_and_timing.loc['CW ' + str(cw), 'rMAE'] = Evaluate_forecast.calc_rmae(path_forecast=path_file,path_real_prices=path_real_prices)
        if 'Time CW ' + str(cw) in dict_timing:
            print(dict_timing)
            Dataframe_evaluation_and_timing.loc['CW ' + str(cw),'Time'] = dict_timing['Time CW ' + str(cw)]
    ensemble_time = sum(Dataframe_evaluation_and_timing['Time'])
    if regular:
        ensemble_file = 'Ensemble_LEAR_forecast_dataframe_' + str(name_dataframe) + '_RW' + str(recalibration_window) + '.csv'
        path_ensemble_file = os.path.join(path_forecasts_folder, ensemble_file)
        Dataframe_evaluation_and_timing.loc['Ensemble',:] = [Evaluate_forecast.calc_mae(path_ensemble_file,path_real_prices=path_real_prices),Evaluate_forecast.calc_rmae(path_ensemble_file,path_real_prices=path_real_prices),ensemble_time]
    if weighed:
        weighed_ensemble_file = 'Weighted_Ensemble_LEAR_forecast_dataframe_' + str(name_dataframe) + '_RW' + str(recalibration_window) + '.csv'
        path_weighed_ensemble_file = os.path.join(path_forecasts_folder, weighed_ensemble_file)
        Dataframe_evaluation_and_timing.loc['Weighed Ensemble',:] = [Evaluate_forecast.calc_mae(path_weighed_ensemble_file,path_real_prices=path_real_prices),Evaluate_forecast.calc_rmae(path_weighed_ensemble_file,path_real_prices=path_real_prices),ensemble_time]


    print('Results for dataframe {}  from {} until {}, RW {}'.format(name_dataframe,begin_test_date, end_test_date,recalibration_window))
    print(Dataframe_evaluation_and_timing)
