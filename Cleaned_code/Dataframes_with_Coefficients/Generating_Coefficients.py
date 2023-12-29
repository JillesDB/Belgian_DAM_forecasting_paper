import datetime

from Epftoolbox_original_code import _lear
import pandas as pd
import os
from Epftoolbox_original_code.evaluation import MAE
from pathlib import Path

Path_cleaned_code = Path.cwd().parent
path_datasets_folder = os.path.join(Path_cleaned_code,'Datasets')
path_real_prices = os.path.join(path_datasets_folder,'Real_prices.csv')
path_forecasts_folder = os.path.join(Path_cleaned_code,'Forecasts_for_plots')

def generate_raw_coefficients(name_dataframe,calibration_window=56,
                                     begin_plot_date=None,end_plot_date=None):
    """

    Parameters
    ----------
    name_dataframe
    calibration_window
    begin_plot_date
    end_plot_date
    group_curves_by
    covariate_family

    Returns
    -------

    """
    hourly_index = pd.date_range(start=begin_plot_date, end=end_plot_date+' 23:00', freq='H')
    dataframe_coefficient = pd.DataFrame(index=pd.DatetimeIndex(hourly_index),columns=range(967))
    #dataframe_coefficient.index.name = 'datetime'

    for count, date in enumerate(pd.date_range(start=begin_plot_date, end=end_plot_date, freq='D')):
        models, effect_matrix, xtest, Yp = (_lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                                 path_recalibration_folder= path_forecasts_folder, dataset=str(name_dataframe), \
                                                                 calibration_window=calibration_window, begin_test_date= (date),
                                                             end_test_date=(date) + datetime.timedelta(hours=23), return_coef_hour=1))
        for h in range(24):
            coef = models[h].coef_
            index = date + datetime.timedelta(hours=h)
            dataframe_coefficient.loc[index] = coef
    name_csv_file = 'Raw_coefficients' + str(name_dataframe) + \
                    '_CW' + str(calibration_window)  + '.csv'
    dataframe_coefficient.to_csv(os.path.join(path_forecasts_folder, name_csv_file),mode='w')



def generate_aggregated_coefficients(name_dataframe,calibration_window=56,
                                     begin_plot_date=None,end_plot_date=None,
                                     group_curves_by = 'date',covariate_family = 'Lagged_Prices'):
    """

    Parameters
    ----------
    name_dataframe
    cw
    begin_plot_date
    end_plot_date
    covariate_family: should be one of the following strings:
    'Lagged_Prices','Wind','Solar','Fossil_Fuels','FR_Generation_Load','Swiss_Prices','BE_Load_Weather'

    Returns
    -------

    """
    #dataframe = pd.read_csv(os.path.join(path_datasets_folder,str(name_dataframe+'.csv')))
    hourly_index = pd.date_range(start=begin_plot_date, end=end_plot_date+' 23:00', freq='H')
    data = {'datetime' : hourly_index , 'Solar' : [0] * len(hourly_index),'Wind' : [0] * len(hourly_index),
            'Lagged_Prices': [0] * len(hourly_index), 'Fossil_Fuels': [0] * len(hourly_index),
            'FR_Generation_Load': [0] * len(hourly_index),'Swiss_Prices': [0] * len(hourly_index),
            'BE_Load_Weather':[0] * len(hourly_index)}
    dataframe_coefficient = pd.DataFrame(data)
    print(hourly_index)

    for count, date in enumerate(pd.date_range(start=begin_plot_date, end=end_plot_date, freq='D')):
        models, effect_matrix, xtest, Yp = (_lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                                 path_recalibration_folder= path_forecasts_folder, dataset=str(name_dataframe), \
                                                                 calibration_window=calibration_window, begin_test_date= (date),
                                                             end_test_date=(date) + datetime.timedelta(hours=23), return_coef_hour=1))
        for h in range(24):
            coef = models[h].coef_
            Abs_Coef_lagged_prices = abs(sum(coef[:96]))
            Abs_Coef_BE_load = abs(sum(coef[96:949:12]))
            Abs_Coef_BE_wind = abs(sum(coef[97:950:12]))
            Abs_Coef_BE_solar = abs(sum(coef[98:951:12]))
            Abs_Coef_CH_price = abs(sum(coef[99:952:12]))
            Abs_Coef_FR_gen = abs(sum(coef[100:953:12]))
            Abs_Coef_DE_wind = abs(sum(coef[101:954:12]))
            Abs_Coef_DE_solar = abs(sum(coef[102:955:12]))
            Abs_Coef_Oil = abs(sum(coef[103:956:12]))
            Abs_Coef_Carbon = abs(sum(coef[104:957:12]))
            Abs_Coef_Gas = abs(sum(coef[105:958:12]))
            Abs_Coef_weather = abs(sum(coef[106:959:12]))
            Abs_Coef_FR_load = abs(sum(coef[107:960:12]))
            #dict_coefficient_values_per_family \
            dataframe_coefficient.iloc[count*24+h,1:]   = [
            (Abs_Coef_BE_solar + Abs_Coef_DE_solar),#'Solar'
            Abs_Coef_BE_wind + Abs_Coef_DE_wind,#'Wind'
            Abs_Coef_lagged_prices,#'Lagged_Prices'
            Abs_Coef_Oil+Abs_Coef_Carbon+Abs_Coef_Gas,#Fossil_Fuels':
            Abs_Coef_FR_gen+Abs_Coef_FR_load,#'FR_Generation_Load'
            Abs_Coef_CH_price,#'Swiss_Prices'
            Abs_Coef_BE_load+Abs_Coef_weather] #'BE_Load_Weather':]
            #dataframe_coefficient.iloc[count*24+h,1] = dict_coefficient_values_per_family[str(covariate_family)]
    name_csv_file = 'Data_clock_plot_dataframe_' + str(name_dataframe) + \
                    '_CW' + str(calibration_window)  + '.csv'
    dataframe_coefficient.to_csv(os.path.join(path_forecasts_folder, name_csv_file),mode='w')

# generate_raw_coefficients(name_dataframe='Full_Dataset', calibration_window=84, begin_plot_date='2020-01-01',
#                       end_plot_date='2022-12-31')
generate_raw_coefficients(name_dataframe='Full_Dataset', calibration_window=721, begin_plot_date='2020-01-01',
                      end_plot_date='2022-12-31')
