import matplotlib.pyplot as plt
from Epftoolbox_original_code.evaluation import _gw,_ancillary_functions,_dm
import pandas as pd
from pathlib import Path
import os
plt.rcParams.update({'font.size': 12})
Path_cleaned_code = Path.cwd().parent
alt.data_transformers.disable_max_rows()
path_datasets_folder = os.path.join(Path_cleaned_code,'Datasets')
path_coeff_folder = os.path.join(Path_cleaned_code,'Dataframes_with_Coefficients')
path_real_prices = os.path.join(path_datasets_folder,'Real_prices.csv')

def DM_plot_forecasts_in_folder(forecasts_folder_path,path_real_prices,plot_title='DM test'):
    """

    Parameters
    ----------
    folder_path: Should contain the path to a folder which has all the forecasts that you want to compare.
    These forecasts should be of format (m x 25), with m the number of days, one column called 'Date' with the
    Date and 24 columns with prices for the price of each hour of that day, named h0 -> h23.

    path_real_prices

    plot_title: The title you want the plot to have.

    Returns
    -------

    """
    dataframe_all_forecasts = None
    for forecast_name in os.listdir(forecasts_folder_path):
        forecast_file_path = os.path.join(forecasts_folder_path,str(forecast_name))
        # checking if it is a file
        if os.path.isfile(forecast_file_path):
            forecast = pd.read_csv(forecast_file_path)
            forecast = forecast.set_index('Date')
            forecast_transformed =  _ancillary_functions._transform_input_prices_for_naive_forecast(forecast,
                                                                                            m='D', freq='1H')
            if dataframe_all_forecasts is None:
                dataframe_all_forecasts = forecast_transformed.copy()
                dataframe_all_forecasts.columns = [*dataframe_all_forecasts.columns[:-1], str(forecast_name)]
            else:
                dataframe_all_forecasts = dataframe_all_forecasts.join(forecast_transformed,rsuffix='new')
                dataframe_all_forecasts.columns = [*dataframe_all_forecasts.columns[:-1], str(forecast_name)]
    dataframe_all_forecasts = dataframe_all_forecasts.reindex(columns = ['CH_Price','2Var_Fam','3Var_Fam','4Var_Fam','5Var_Fam','Full_Dataset'])
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    real_prices_transformed = _ancillary_functions._transform_input_prices_for_naive_forecast(real_prices, m='D', freq='1H')
    real_prices_selection = real_prices_transformed.loc[dataframe_all_forecasts.index].copy()
    _dm.plot_multivariate_DM_test(real_price=real_prices_selection, forecasts=dataframe_all_forecasts,
                              title=plot_title)


def GW_plot_forecasts_in_folder(forecasts_folder_path,path_real_prices, plot_title='GW test'):
    """

    Parameters
    ----------
    folder_path: Should contain the path to a folder which has all the forecasts that you want to compare.
    These forecasts should be of format (m x 25), with m the number of days, one column called 'Date' with the
    Date and 24 columns with prices for the price of each hour of that day, named h0 -> h23.

    path_real_prices

    plot_title: The title you want the plot to have.

    Returns
    -------

    """
    dataframe_all_forecasts = None
    for forecast_name in os.listdir(forecasts_folder_path):
        forecast_file_path = os.path.join(forecasts_folder_path,str(forecast_name))
        if os.path.isfile(forecast_file_path):
            forecast = pd.read_csv(forecast_file_path)
            forecast = forecast.set_index('Date')
            forecast_transformed =  _ancillary_functions._transform_input_prices_for_naive_forecast(forecast,
                                                                                            m='D', freq='1H')
            if dataframe_all_forecasts is None:
                dataframe_all_forecasts = forecast_transformed.copy()
                dataframe_all_forecasts.columns = [*dataframe_all_forecasts.columns[:-1], str(forecast_name)]
            else:
                dataframe_all_forecasts = dataframe_all_forecasts.join(forecast_transformed,rsuffix='new')
                dataframe_all_forecasts.columns = [*dataframe_all_forecasts.columns[:-1], str(forecast_name)]
    dataframe_all_forecasts = dataframe_all_forecasts.reindex(columns = ['CH_Price','2Var_Fam','3Var_Fam','4Var_Fam','5Var_Fam','Full_Dataset'])
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    real_prices_transformed = _ancillary_functions._transform_input_prices_for_naive_forecast(real_prices, m='D', freq='1H')
    real_prices_selection = real_prices_transformed.loc[dataframe_all_forecasts.index].copy()
    print(real_prices_selection['Prices'],dataframe_all_forecasts)
    _gw.plot_multivariate_GW_test(real_price=real_prices_selection['Prices'], forecasts=dataframe_all_forecasts,
                              title=plot_title)

# DM_plot_forecasts_in_folder(forecasts_folder_path=path_forecasts_folder,path_real_prices=path_real_prices, plot_title='DM test')
