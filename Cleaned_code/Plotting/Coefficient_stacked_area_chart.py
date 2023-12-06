import pandas as pd

from Epftoolbox_original_code import _lear
import datetime
import altair as alt
import os
from pathlib import Path
import altair_viewer
alt.data_transformers.disable_max_rows()

import numpy as np

path_datasets_folder = str(Path.cwd().parent) + '\Datasets'
path_forecasts_folder = str(Path.cwd().parent) + '\Dataframes_with_Coefficients'
real_prices = pd.read_csv(os.path.join(path_datasets_folder,'Real_prices.csv'))
real_prices = real_prices.set_index('Date')
def generate_coef_analysis_dict(name_dataframe,begin_plot_date,end_plot_date,cal_window,hour=0):

    h =hour
    # dataframe = pd.read_csv(os.path.join(path_datasets_folder,str(name_dataframe+'.csv')))
    # hourly_index = pd.date_range(start=begin_plot_date, end=end_plot_date + ' 23:00', freq='H')
    # data = {'datetime': hourly_index, 'Solar': [0] * len(hourly_index), 'Wind': [0] * len(hourly_index),
    #         'Lagged_Prices': [0] * len(hourly_index), 'Fossil_Fuels': [0] * len(hourly_index),
    #         'FR_Generation_Load': [0] * len(hourly_index), 'Swiss_Prices': [0] * len(hourly_index),
    #         'BE_Load_Weather': [0] * len(hourly_index)}
    dict_coefficient_values_per_family = []
    print(name_dataframe)
    for count, date in enumerate(pd.date_range(start=begin_plot_date, end=end_plot_date, freq='D')):
        models, effect_matrix, xtest, Yp = (
            _lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                path_recalibration_folder=path_forecasts_folder,
                                                dataset=str(name_dataframe), \
                                                calibration_window=cal_window, begin_test_date=(date),
                                                end_test_date=(date) + datetime.timedelta(hours=23),
                                                return_coef_hour=1))
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

        dict_coefficient_values_per_family.extend([
          {"day": date.strftime("%d-%b-%Y"),
            "Abs_Coef": "Abs. Coeff. Lagged Prices",
            "value": Abs_Coef_lagged_prices},
          {"day": date.strftime("%d-%b-%Y"),
           "Abs_Coef": "Abs. Coeff. BE Load & Weather",
           "value": Abs_Coef_BE_load+Abs_Coef_weather},
          {"day": date.strftime("%d-%b-%Y"),
           "Abs_Coef": "Abs. Coeff. Wind Forecast",
           "value": Abs_Coef_DE_wind+Abs_Coef_BE_wind},
          {"day": date.strftime("%d-%b-%Y"),
           "Abs_Coef": "Abs. Coeff. Solar Forecast",
           "value": Abs_Coef_DE_solar+Abs_Coef_BE_solar},
          {"day": date.strftime("%d-%b-%Y"),
           "Abs_Coef": "Abs. Coeff. Fossil Fuels",
           "value": Abs_Coef_Oil+Abs_Coef_Carbon+Abs_Coef_Gas},
          {"day": date.strftime("%d-%b-%Y"),
           "Abs_Coef": "Abs. Coeff. CH Prices",
           "value": Abs_Coef_CH_price},
            {"day": date.strftime("%d-%b-%Y"),
             "Abs_Coef": "Abs. Coeff. FR Load & Generation",
         "value": Abs_Coef_FR_load+Abs_Coef_FR_gen}])
        return dict_coefficient_values_per_family
    # path_datasets_folder = str(Path.cwd().parent) + '\Datasets'
    # path_forecasts_folder = str(Path.cwd().parent)+'\Coefficients_for_clock_plots'
    # day = datetime.date(2020,1,1) +  datetime.timedelta(day_nr)
    # start = datetime.datetime.strptime("01/01/2020 00:00","%d/%m/%Y %H:%M") + datetime.timedelta(day_nr)
    # end = datetime.datetime.strptime("01/01/2020 23:00","%d/%m/%Y %H:%M") + datetime.timedelta(day_nr)
    # print(start,end)
    # models, effect_matrix, xtest, Yp = (_lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
    #                                                                         path_recalibration_folder=path_forecasts_folder,
    #                                                                         dataset=str(name_dataframe), \
    #                                                                         calibration_window=cw,
    #                                                                         begin_test_date=(day_nr),
    #                                                                         end_test_date=(day_nr) + datetime.timedelta(
    #                                                                             hours=23), return_coef_hour=1))
    # #print(effect_matrix,xtest)
    # zeros = effect_matrix[14]
    # real_prices_day = real_prices.iloc[day_nr,:]
    # intercepts = effect_matrix[15]
    # std_prices = effect_matrix[0]
    # coef_dict_day = []
    # coef = models[h].coef_
    # Abs_Coef_lag_prices = abs(sum(coef[:96]))
    # Abs_Coef_BE_load = abs(sum(coef[96:949:12]))
    # Abs_Coef_BE_wind= abs(sum(coef[97:950:12]))
    # Abs_Coef_BE_solar= abs(sum(coef[98:951:12]))
    # Abs_Coef_CH_price= abs(sum(coef[99:952:12]))
    # Abs_Coef_FR_gen= abs(sum(coef[100:953:12]))
    # Abs_Coef_DE_wind= abs(sum(coef[101:954:12]))
    # Abs_Coef_DE_solar= abs(sum(coef[102:955:12]))
    # Abs_Coef_Oil = abs(sum(coef[103:956:12]))
    # Abs_Coef_Carbon = abs(sum(coef[104:957:12]))
    # Abs_Coef_Gas= abs(sum(coef[105:958:12]))
    # Abs_Coef_weather= abs(sum(coef[106:959:12]))
    # Abs_Coef_FR_load= abs(sum(coef[107:960:12]))
    # sum_abs_coeff = (
    #         Abs_Coef_lag_prices
    #         + Abs_Coef_BE_load
    #         + Abs_Coef_BE_wind
    #         + Abs_Coef_BE_solar
    #         + Abs_Coef_CH_price
    #         + Abs_Coef_FR_gen
    #         + Abs_Coef_DE_wind
    #         + Abs_Coef_DE_solar
    #         + Abs_Coef_Oil
    #         + Abs_Coef_Carbon
    #         + Abs_Coef_Gas
    #         + Abs_Coef_weather
    #         + Abs_Coef_FR_load
    # )
    # coef_dict_day.extend([
    #   {"day": day.strftime("%d-%b-%Y"),
    #     "Abs_Coef": "Abs. Coeff. Lagged Prices",
    #     "value": Abs_Coef_lag_prices},
    #   {"day": day.strftime("%d-%b-%Y"),
    #    "Abs_Coef": "Abs. Coeff. BE Load & Weather",
    #    "value": Abs_Coef_BE_load+Abs_Coef_weather},
    #   {"day": day.strftime("%d-%b-%Y"),
    #    "Abs_Coef": "Abs. Coeff. Wind Forecast",
    #    "value": Abs_Coef_DE_wind+Abs_Coef_BE_wind},
    #   {"day": day.strftime("%d-%b-%Y"),
    #    "Abs_Coef": "Abs. Coeff. Solar Forecast",
    #    "value": Abs_Coef_DE_solar+Abs_Coef_BE_solar},
    #   {"day": day.strftime("%d-%b-%Y"),
    #    "Abs_Coef": "Abs. Coeff. Fossil Fuels",
    #    "value": Abs_Coef_Oil+Abs_Coef_Carbon+Abs_Coef_Gas},
    #   {"day": day.strftime("%d-%b-%Y"),
    #    "Abs_Coef": "Abs. Coeff. CH Prices",
    #    "value": Abs_Coef_CH_price},
    #     {"day": day.strftime("%d-%b-%Y"),
    #      "Abs_Coef": "Abs. Coeff. FR Load & Generation",
    #  "value": Abs_Coef_FR_load+Abs_Coef_FR_gen}])
    # return coef_dict_day,day,real_prices_day,std_prices


def create_stacked_area_chart(begin_plot_date,end_plot_date,name_csv='None',name_dataframe='None',hour=None,cal_window=56):
    """

    Parameters
    ----------
    hour
    file_path
    cal_window: should only be given if no file path given
    dates

    Returns
    -------

    """
    file_path = os.path.join(path_forecasts_folder,str(name_csv)+'.csv')
    print(file_path)
    if not os.path.exists(file_path):
        dataframe_coefficients = generate_coef_analysis_dict(name_dataframe=name_dataframe,begin_plot_date=begin_plot_date,
                                    end_plot_date=end_plot_date,cal_window=cal_window,hour=hour)
        dataframe  = alt.Data(values = dataframe_coefficients)
    else:
        if hour != None:
            print(file_path)
            dataframe = pd.read_csv(str(file_path))
            day_selection = (dataframe['datetime'] >= begin_plot_date) & (dataframe['datetime'] <= end_plot_date+' 23:00:00')
            dataframe = dataframe.loc[day_selection]
            dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
            dataframe = dataframe.set_index('datetime')
            dataframe = dataframe.drop(columns=['Unnamed: 0'],axis=1)
            dataframe = dataframe[dataframe.index.hour == hour]
            #dataframe.index = dataframe.index.date
            dataframe.index.name = 'Date'
            dataframe = dataframe.reset_index(drop = False)
            dataframe = dataframe.melt(id_vars='Date', var_name='Var_Family', value_name='value')
        else:
            dataframe = pd.read_csv(file_path)
            day_selection = (dataframe['datetime'] >= begin_plot_date) & (dataframe['datetime'] <= end_plot_date)  # + datetime.timedelta(hours=23))
            dataframe = dataframe.loc[day_selection]
            dataframe['Date'] = pd.to_datetime(dataframe['datetime'])
            dataframe = dataframe.drop(columns=['Unnamed: 0','datetime'],axis=1)
            dataframe = dataframe.melt(id_vars='Date', var_name='Var_Family', value_name='value')

        print(dataframe)
    #dataframe_coefficients_long_form = dataframe.melt(id_vars=["datetime"],
    #                                                            value_vars=["Solar", "Wind", "Lagged_Prices",
    #                                                                            "Fossil_Fuels", "FR_Generation_Load",
    #                                                                          'Swiss_Prices', 'BE_Load_Weather'])
    #print(dataframe_coefficients_long_form)
    #data1  = alt.Data(values = dataframe)
    #print(data1)
    bar_chart = alt.Chart(dataframe).mark_area().encode(
        x=alt.X('Date:T'),# axis=alt.Axis(title='Date',ticks = True,format="%b %Y")),
        y=alt.Y('value:Q', axis=alt.Axis(title='Absolute Value Coefficients')),
        color=alt.Color('Var_Family:N',scale=alt.Scale(range=["#c7ead4", "#b4e0aa", "#c5e08b", "#e5e079", "#f6d264", "#f5b34c", "#f4913e"]))#["#c4e9d0", "#b0de9f", "#d0e181", "#e5e079", "#f6e072", "#f6c053", "#f3993e"]
    )
    # ##31688e
    # combine the bar chart, text and point layers
    chart = alt.layer(bar_chart).properties(title='Evolution of Absolute Coefficients - CW728 on Hour '+str(hour)).resolve_scale(color='independent')
    #chart = alt.layer(bar_chart,point,text2).properties(title='Prices for ' + str(day)).resolve_scale(color='independent')
    # display the chart
    chart.show()
dates_nr = range(728)
#create_stacked_area_chart(hour=18,file_path=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Coefficients_for_clock_plots\Data_clock_plot_dataframe_Example_dataframe_CW728.csv')
create_stacked_area_chart(name_csv = 'Data_clock_plot_dataframe_Example_dataframe_CW728',
                          begin_plot_date='2020-01-01',end_plot_date='2021-01-01')
create_stacked_area_chart(name_csv = 'Data_clock_plot_dataframe_Example_dataframe_CW728',hour=12,
                      begin_plot_date='2020-01-01',end_plot_date='2020-02-01')