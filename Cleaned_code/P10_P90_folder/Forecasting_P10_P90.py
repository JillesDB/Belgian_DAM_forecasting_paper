import _lear_P10_P90
import pandas as pd
import os
from Epftoolbox_original_code.evaluation._mae import MAE
import time
import Generate_Evaluate_and_Time_Forecast_P10_P90
from pathlib import Path

cwd = Path.cwd()
cleaned_code_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(cwd, 'Datasets/P10_P90_datasets')
forecasts_folder = os.path.join(cwd, 'Forecasts2')
real_prices_path = os.path.join(cwd, 'Datasets\Real_prices.csv')


def create_single_forecast(dataset_train,dataset_test,path_forecast_folder=None,calibration_window=56,begin_test_date=None,end_test_date=None,recalibration_window=1):
    """ 1) The Dataframes:
     The dataset folder should contain the dataset. The folder should be specified in the Dataframes folder
     The dataset should be a csv file with a pandas dataframe, with the 'Date' column as index. The 'Date' index column should specify the date as 'YYYY-MM-DD HH:MM'
     Similarly, in the path recalibration folder, where the dataframe with forecasted prices will be stored, should also be specified as r'C:\path_to_store_the_forecast'
     Begin date:    Parameter to determine the begin date for test period. It should be a string with format YYYY-mm-dd

     The end date:  Parameter to determine the end date for test period. It should be a string with format YYYY-mm-dd
    """

    if path_forecast_folder is None:
        path_forecast_folder = str(cwd) + '\Forecasts'
    _lear_P10_P90.evaluate_lear_in_test_dataset(path_datasets_folder=datasets_path,path_recalibration_folder=path_forecast_folder, dataset_train=str(dataset_train), dataset_test=str(dataset_test), \
                                        calibration_window=calibration_window, begin_test_date=str(begin_test_date) + ' 00:00',
                                        end_test_date=str(end_test_date) + ' 23:00', recalibration_window=recalibration_window)

def create_ensemble_forecast(dataset_train,dataset_test,path_real_prices,path_datasets_folder,path_forecasts_folder=None,
                             begin_test_date=None,end_test_date=None,recalibration_window=1,
                             calibration_window_set=tuple([56,84,112,714,721,728]),weighed=0,regular=0,return_time=0):
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
    weighed should be =1 to generate the weighed ensemble
    regular should be =1 to generate the non-weighed ensemble
    the weights for the weighted ensemble are each time define by the performance of each of the predictions
    in the last 24 hours


    Returns the Ensemble or the weighted Ensemble, as a pandas dataframe with the date as index,
    in the format YYYY-MM-DD, and with 24 columns named h0 - h23 with the forecast prices for each
    of the 24 hours for that day.
    -------

    """

    list_forecasts = []
    real_prices = pd.read_csv(path_real_prices)
    real_prices['Date'] = pd.to_datetime(real_prices['Date'])
    real_prices = real_prices.set_index('Date')
    if path_forecasts_folder is None:
        path_forecasts_folder = str(cwd)+'\Forecasts'
    timing = dict()
    for cw in calibration_window_set:
        name_csv_file = 'LEAR_train_dataframe_' + str(dataset_train) +'test_dataframe_' + str(dataset_test) + \
                         '_CW' + str(cw) + '_RW' + str(recalibration_window) + '.csv'
        path_file = os.path.join(path_forecasts_folder,name_csv_file)
        'check whether forecast exists already'
        if os.path.exists(path_file):
            a = pd.read_csv(path_file)
            a = a.set_index('Date')
            if (str(pd.to_datetime(begin_test_date).date()) not in a.index or str(pd.to_datetime(end_test_date).date()) not in a.index) or (a.loc[str(pd.to_datetime(begin_test_date).date())].isna().any() or a.loc[str(pd.to_datetime(end_test_date).date())].isna().any()):
                print('forecasting file ' + str(path_file))
                start1 = time.time()
                a = _lear_P10_P90.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                        path_recalibration_folder=path_forecasts_folder, dataset_train=str(dataset_train), dataset_test=str(dataset_test), \
                                                        calibration_window=cw,
                                                        begin_test_date=str(begin_test_date) + ' 00:00',
                                                        end_test_date=str(end_test_date) + ' 23:00',recalibration_window=recalibration_window,timing=0)
                # print('time to forecast dataframe ' + str(name_dataframe) + ' CW ' + str(cw)
                #       + ' from ' + str(begin_test_date) + ' until ' + str(end_test_date) + ' took ' + str(
                #     time.time() - start1))
                timing['Time CW '+str(cw)] = time.time() - start1
        else:
            start2 = time.time()
            a = _lear_P10_P90.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                            path_recalibration_folder=path_forecasts_folder,dataset_train=str(dataset_train),dataset_test=str(dataset_test), \
                                                            calibration_window=cw,
                                                    begin_test_date=str(begin_test_date) + ' 00:00',
                                                    end_test_date=str(end_test_date) + ' 23:00',
                                                    recalibration_window=recalibration_window,timing=0)
            # print('time to forecast dataframe ' + str(name_dataframe)+' CW ' + str(cw)
            #       +' from '+ str(begin_test_date) + ' until ' + str(end_test_date) + ' took ' + str(time.time() - start2))
            timing['Time CW ' + str(cw)] = time.time() - start2
            #print(a)
        a.index = pd.to_datetime(a.index)
        #a.index = pd.Series(a.index.format())
        list_forecasts.append(a)
    print(timing)
    if weighed == 1:
        Weighted_Ensemble_file_name = 'Weighted_Ensemble_LEAR_train' + str(dataset_train) +'test_dataframe_' + str(dataset_test) + \
                                        '_RW' + str(recalibration_window) + '.csv'

        Weighted_Ensemble_file_path = os.path.join(path_forecasts_folder,Weighted_Ensemble_file_name)


        Weighted_Ensemble = pd.DataFrame(0,index=list_forecasts[0].index, columns=list_forecasts[0].columns)
        #print(real_prices.index.dtype,list_forecasts[0].index.dtype)
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
    if regular ==1:
        Ensemble_file_name = 'Ensemble_LEAR_train' + str(dataset_train) +'test_dataframe_' + str(dataset_test) + \
                         '_RW' + str(recalibration_window) + '.csv'

        Ensemble_file_path = os.path.join(path_forecasts_folder,Ensemble_file_name)
        Ensemble = pd.DataFrame(0,index=list_forecasts[0].index, columns=list_forecasts[0].columns)

        for i in Ensemble.index:
            for forecast_number in range(len(list_forecasts)):
                for j in Ensemble.columns:
                    Ensemble.loc[i, j] += (list_forecasts[forecast_number].loc[i, j]) / len(list_forecasts)
        Ensemble.to_csv(Ensemble_file_path)
    if return_time:
        timing['Ensemble/Weighted Ensemble'] = sum(timing.values())
        return timing


def create_ensemble_one_day(list_forecasts,real_prices,day,weighted=0,regular=0):
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


    if regular == 1:

        Ensemble_Forecast = pd.DataFrame(0,columns=real_prices.columns,index=day)
        for forecast_number in range(len(list_forecasts)):

                    Ensemble_Forecast.loc[day] += (list_forecasts[forecast_number].iloc[i, 1:]/ len(list_forecasts))



# create_ensemble_forecast(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
#                          begin_test_date='2021-01-01 00:00', end_test_date='2021-01-31 23:00', weighed=1)
# create_ensemble_forecast(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
#                          begin_test_date='2021-01-01 00:00', end_test_date='2021-01-31 23:00')
