from Epftoolbox_original_code import _lear
import pandas as pd
import os
from evaluation import MAE
from pathlib import Path

cwd = Path.cwd()
def create_single_forecast(name_dataframe,calibration_window=56,begin_test_date=None,end_test_date=None,recalibration_window=1,years_test=0):
    """ 1) The Dataframes:
     The dataset folder should contain the dataset. The folder should be specified in the Dataframes folder
     The dataset should be a csv file with a pandas dataframe, with the 'Date' column as index. The 'Date' index column should specify the date as 'YYYY-MM-DD HH:MM'
     Similarly, in the path recalibration folder, where the dataframe with forecasted prices will be stored, should also be specified as r'C:\path_to_store_the_forecast'
     Begin date:    Should be specified as DD/MM/YYYY 00:00 (00:00 referring to the values for the first hour of day,
                    from 00:00-01:00).

     The end date: Optional parameter to select the test dataset. Used in combination with ' +
                         'begin_test_date. If either of them is not provided, test dataset is built ' +
                         'using the years_test parameter. It should either be  a string with the ' +
                         ' following format d/m/Y H:M"""
    path_datasets_folder = "r\'" + str(cwd)+'\Datasets' + "\'"
    path_forecasts_folder = "r\'" + str(cwd)+'\Forecasts' + "\'"
    _lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                        path_recalibration_folder=path_forecasts_folder, dataset=str(name_dataframe), years_test=years_test, \
                                        calibration_window=calibration_window, begin_test_date=str(begin_test_date), end_test_date=str(end_test_date), recal_interval=recalibration_window)

def create_ensemble_forecast(name_dataframe,path_real_prices,begin_test_date=None,end_test_date=None,recalibration_window=1,
                             set_cws=frozenset([56,84,112,714,721,728]),weighed=0,years_test=0):
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

    list_forecasts = []
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    path_datasets_folder = str(cwd)+'\Datasets'
    path_forecasts_folder = str(cwd)+'\Forecasts'
    for cw in set_cws:
        name_csv_file = 'LEAR_forecast' + '_dat' + str(name_dataframe) + '_YT' + str(years_test) + \
                         '_CW' + str(cw) + '_RW' + str(recalibration_window) + '.csv'
        path_file = os.path.join(path_forecasts_folder,name_csv_file)
        'check whether forecast exists already'
        if os.path.exists(path_file):
            a = pd.read_csv(path_file)
            a = a.set_index('Date')
            if a.loc[str(pd.to_datetime(begin_test_date).date())].isna().any() or a.loc[str(pd.to_datetime(end_test_date).date())].isna().any():
                print('forecasting file ' + str(path_file))
                a = _lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                        path_recalibration_folder=path_forecasts_folder, dataset=str(name_dataframe), \
                                                        calibration_window=cw,
                                                        begin_test_date=str(begin_test_date) + ' 00:00',
                                                        end_test_date=str(end_test_date) + ' 23:00',
                                                        recal_interval=recalibration_window, years_test=years_test)
        else:
            a = _lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                    path_recalibration_folder=path_forecasts_folder, dataset=str(name_dataframe), \
                                                    calibration_window=cw,
                                                    begin_test_date=str(begin_test_date) + ' 00:00',
                                                    end_test_date=str(end_test_date) + ' 23:00',
                                                    recal_interval=recalibration_window, years_test=years_test)
        list_forecasts.append(a)
    if weighed != 0:
        Weighted_Ensemble_file_name = 'Weighted_Ensemble_LEAR_forecast' + '_dat' + str(name_dataframe) \
                                      + '_YT' + str(years_test) + '_RW' + str(recalibration_window) + '.csv'

        Weighted_Ensemble_file_path = os.path.join(path_forecasts_folder,Weighted_Ensemble_file_name)


        Weighted_Ensemble = pd.DataFrame(0,index=list_forecasts[0].index, columns=list_forecasts[0].columns)
        print(Weighted_Ensemble,real_prices)
        real_prices_selection = real_prices.loc[list_forecasts[0].index].copy()
        for i in range(len(Weighted_Ensemble.index)):
            if i == 0:
                for forecast_number in range(len(list_forecasts)):
                    b = (list_forecasts[forecast_number].iloc[0])
                    Weighted_Ensemble.iloc[0] = Weighted_Ensemble.iloc[0] + b
                Weighted_Ensemble.iloc[0] = Weighted_Ensemble.iloc[0].div(len(list_forecasts))
            else:
                #print(Weighted_Ensemble)
                s=0
                for forecast_number in range(len(list_forecasts)):
                    s += (1 / MAE(real_prices_selection.iloc[i - 1, :].values.squeeze(),list_forecasts[forecast_number].iloc[i - 1,:].values.squeeze()))
                for forecast_number in range(len(list_forecasts)):
                    b= 1 / (s * MAE(real_prices_selection.iloc[i - 1, :].values.squeeze(),list_forecasts[forecast_number].iloc[i - 1,:].values.squeeze())) * list_forecasts[forecast_number].iloc[i, :]
                    #print(b)
                    Weighted_Ensemble.iloc[i, :] += b
        Weighted_Ensemble.to_csv(Weighted_Ensemble_file_path)
        return Weighted_Ensemble

    else:
        Ensemble_file_name = 'Ensemble_LEAR_forecast' + '_dat' + str(name_dataframe) + '_YT' + str(years_test) + \
                         '_RW' + str(recalibration_window) + '.csv'

        Ensemble_file_path = os.path.join(path_forecasts_folder,Ensemble_file_name)
        Ensemble = pd.DataFrame(0,index=list_forecasts[0].index, columns=list_forecasts[0].columns)

        for forecast_number in range(len(list_forecasts)):
            for j in Ensemble.columns:
                for i in Ensemble.index:
                    Ensemble.loc[i, j] += (list_forecasts[forecast_number].loc[i, j]) / len(list_forecasts)
        Ensemble.to_csv(Ensemble_file_path)
        return Ensemble

def create_ensemble_one_day(list_forecasts,real_prices,day,weighted=0):
    """

    Parameters
    ----------
    list_forecasts should contain a list of all forecasts that are to be combined in the ensemble.
    Each of the forecasts should be pandas datasets, with the 'Date' column (format 'YYYY-MM-DD')
    plus 24 columns for the forecast of each of the 24 hours of that day.
    real_prices
    the day should, like the 'Date' index columns of real_prices and list_forecasts,
    specify the date for which the ensemble is to be calculated as 'YYYY-MM-DD'

    Returns
    -------

    """
    i = real_prices.index[real_prices['Date']==day]
    real_prices  = real_prices.set_index('Date')
    if weighted != 0:
        Weighted_Ensemble_Forecast = pd.DataFrame(0,columns=real_prices.columns,index=day)
        s=0
        for forecast_number in range(len(list_forecasts)):
            s += (1 / MAE(real_prices.iloc[i - 1, :].values.squeeze(),list_forecasts[forecast_number].iloc[i - 1,0:].values.squeeze()))
        for forecast_number in range(len(list_forecasts)):
            Weighted_Ensemble_Forecast.loc[day] += (1 / (s * MAE(real_prices.iloc[i - 1, :].values.squeeze(),list_forecasts[forecast_number].iloc[i - 1,0:].values.squeeze())) * list_forecasts[forecast_number].iloc[i, :])

        return Weighted_Ensemble_Forecast

    else:

        Ensemble_Forecast = pd.DataFrame(0,columns=real_prices.columns,index=day)
        for forecast_number in range(len(list_forecasts)):

                    Ensemble_Forecast.loc[day] += (list_forecasts[forecast_number].iloc[i, 1:]/ len(list_forecasts))

create_ensemble_forecast(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
                         begin_test_date='2021-01-01 00:00', end_test_date='2021-01-31 23:00', weighed=1)
create_ensemble_forecast(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
                         begin_test_date='2021-01-01 00:00', end_test_date='2021-01-31 23:00')
