from Epftoolbox_original_code import _lear
import pandas as pd
import os
from evaluation import MAE
from pathlib import Path

cwd = Path.cwd()
def time_forecast(name_dataframe,path_real_prices=None,begin_test_date=None,end_test_date=None,recalibration_window=1,
                             set_cws=frozenset([56,84,112,714,721,728]),years_test=0):
    """



    Parameters
    ----------
    begin_test_date
    end_test_date
    name_dataframe
    recalibration_window
    To create the weighted path_real_prices should define the path of a csv file. This csv file should contain
    a pandas dataframe with at least the following two colums: the first column, called 'Date',should have the date
    in the format of 'YYYY-MM-DD HH:MM'. The second column should contain the prices.
    set_cws should contain a set {} with all calibration windows that will make up the ensemble/weighted ensemble
    weighed should be =1 for a weighed ensemble
    years_test can be used as an alternative to begin/end_test_date.
    the weights for the weighted ensemble are each time define by the performance of each of the predictions
    in the last 24 hours


    Returns the Ensemble or the weighted Ensemble, as a pandas dataframe with the date as index,
    in the format YYYY-MM-DD, and with 24 columns named h0 - h23 with the forecast prices for each
    of the 24 hours for that day.
    -------

    """

    timing_forecasts = pd.DataFrame()
    path_datasets_folder = str(cwd)+'\Datasets'
    path_forecasts_folder = str(cwd)+'\Forecasts'
    for cw in set_cws:
        name_csv_file = 'LEAR_forecast' + 'timing' + str(name_dataframe) + '_YT' + str(years_test) + \
                         '_CW' + str(cw) + '_RW' + str(recalibration_window) + '.csv'
        path_file = os.path.join(path_forecasts_folder,name_csv_file)
        'check whether forecast exists already'
        print('forecasting file ' + str(path_file))
        a = _lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                        path_recalibration_folder=path_forecasts_folder, dataset=str(name_dataframe), \
                                                        calibration_window=cw,
                                                        begin_test_date=str(begin_test_date) + ' 00:00',
                                                        end_test_date=str(end_test_date) + ' 23:00',
                                                        recal_interval=recalibration_window, years_test=years_test,
                                                        timing=1)
        timing_forecasts = pd.concat([timing_forecasts,a],axis=1)
    timing_forecasts.loc['average computation time'] = timing_forecasts.mean()
    return timing_forecasts.loc['average computation time']

b = time_forecast(name_dataframe='Example_dataframe',begin_test_date='01-01-2021 00:00',
              end_test_date='31/03/2021 23:00', set_cws={56,84,728})
print(b)