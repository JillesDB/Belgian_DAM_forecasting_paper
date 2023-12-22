from Epftoolbox_original_code.evaluation._mae import MAE
from Epftoolbox_original_code.evaluation._rmae import rMAE
import numpy as np
import pandas as pd
import os
from pathlib import Path

cwd = Path.cwd()
path_forecasts_folder = os.path.join(cwd,'Forecasts')
path_datasets_folder = os.path.join(cwd,'Datasets')
path_real_prices = os.path.join(path_datasets_folder,'Real_prices.csv')



def calc_mae(path_forecast,path_real_prices,begin_test_date=None,end_test_date=None):
    """

    Parameters
    ----------
    file_forecast: the name of your dataframe, which should be in the Forecasts directory
    path_real_prices should be in the form r'your_path'
    begin_test_date
    end_test_date

    Returns
    -------

    """
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    if os.path.isfile(path_forecast):
        forecast = pd.read_csv(path_forecast)
        #print(forecast,real_prices)
        forecast = forecast.set_index('Date')
        real_prices_indexed = real_prices.loc[forecast.index].copy()
        MAE_forecast = np.mean(MAE(real_prices_indexed, forecast))
        print('The MAE for this forecast is '+ str(MAE_forecast))
        return MAE_forecast
    else:
        print('forecast path not found')


def calc_rmae(path_forecast, path_real_prices, begin_test_date=None, end_test_date=None,m = 'W', freq='1H'):
    """

    Parameters
    ----------
    name_dateframe
    path_real_prices should be in the form r'your_path'
    begin_test_date
    end_test_date
    path_recalibration_folder

    Returns
    -------

    """
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    if os.path.isfile(path_forecast):
        forecast = pd.read_csv(path_forecast)
        forecast = forecast.set_index('Date')
        real_prices_indexed = real_prices.loc[forecast.index].copy()
        rMAE_forecast = np.mean(rMAE(real_prices_indexed, forecast, m='D', freq='1H'))
        print('The rMAE for this forecast is '+ str(rMAE_forecast))
        return rMAE_forecast
    else:
        print('forecast path not found')


# calc_rmae(path_forecast=os.path.join(path_forecasts_folder,'Weighted_Ensemble_LEAR_forecast_datExample_dataframe_YT0_RW1'), path_real_prices=path_real_prices)
