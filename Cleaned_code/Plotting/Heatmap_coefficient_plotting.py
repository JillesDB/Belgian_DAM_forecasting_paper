import datetime

from Epftoolbox_original_code import _lear
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date,time
sns.set(rc={'figure.figsize':(6,36)})
path_datasets_folder = str(Path.cwd().parent) + '\Dataframes_with_coefficients'
path_forecasts_folder = str(Path.cwd().parent) + '\Forecasts_for_plots'
path_real_prices = os.path.join(path_datasets_folder,'Real_prices.csv')

def plot_coefficient_matrix_heatmap(name_csv,calibration_window,begin_time=None,end_time=None):
    """

    Parameters
    ----------
    path_file

    Returns
    -------

    """
    path_file = os.path.join(path_datasets_folder,name_csv)
    dataframe_coefficients = pd.read_csv(path_file)
    if begin_time != None:
        day_selection = (dataframe_coefficients['datetime'] >= begin_time) & (dataframe_coefficients['datetime'] <= end_time)# + datetime.timedelta(hours=23))
        filtered_dataframe = dataframe_coefficients.loc[day_selection]
    else:
        filtered_dataframe = dataframe_coefficients
    print(filtered_dataframe)
    filtered_dataframe = filtered_dataframe.set_index(pd.DatetimeIndex(filtered_dataframe['datetime'])).drop(columns=['Unnamed: 0','datetime'])
    filtered_dataframe_day = filtered_dataframe.between_time('06:30','22:30') #16 hours per day
    filtered_dataframe_night = filtered_dataframe.between_time('22:30','06:30')#8 hours per night
    # print(filtered_dataframe_day,filtered_dataframe,filtered_dataframe_night)
    # filtered_dataframe = filtered_dataframe.apply(lambda y: (y - y.mean()) / y.std(), axis=0)
    # filtered_dataframe_day = filtered_dataframe_day.apply(lambda y: (y - y.mean()) / y.std(), axis=0)
    # filtered_dataframe_night = filtered_dataframe_night.apply(lambda y: (y - y.mean()) / y.std(), axis=0)
    nb_row = 16*7*4 #15*7 hours per week => four week average
    filtered_dataframe_day = filtered_dataframe_day.rolling(nb_row).mean()[nb_row-1:]
    filtered_dataframe = filtered_dataframe.rolling(nb_row).mean()[nb_row-1:]
    filtered_dataframe_night = filtered_dataframe_night.rolling(nb_row).mean()[nb_row-1:]
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(filtered_dataframe_day,yticklabels=nb_row)# yticklabels = 8760/nb_row
    ax.set_title('Coefficients for CW' + str(calibration_window)+' from '+str(filtered_dataframe.index[0]) + ' until ' +str(filtered_dataframe.index[-1]))
    plt.tight_layout()
    plt.show()

# plot_coefficient_matrix_heatmap(name_csv='Aggregated_Coefficients_Full_Dataset_CW56.csv',calibration_window=56,
#                                 begin_time='2020-01-01 00:00:00',end_time='2021-01-01 23:00:00')
