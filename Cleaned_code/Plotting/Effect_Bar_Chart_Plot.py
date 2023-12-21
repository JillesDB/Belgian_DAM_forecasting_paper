import pandas as pd
from Epftoolbox_original_code import _lear_std
import datetime
from pathlib import Path
import os
import altair as alt
import numpy as np
import matplotlib.pyplot as plt

cwd = Path.cwd().parent
alt.data_transformers.disable_max_rows()
path_coefficients_folder = os.path.join(cwd,'Dataframes_with_Coefficients')
path_datasets_folder = os.path.join(cwd,'Datasets')
path_real_prices = (os.path.join(path_datasets_folder,'Real_prices.csv'))
real_prices = pd.read_csv(path_real_prices)
real_prices = real_prices.set_index('Date')
def generate_coef_analysis_dict(day_nr,cw,hour=0):
    h =hour
    day = datetime.date(2021,1,1) +  datetime.timedelta(day_nr)
    start = datetime.datetime.strptime("01/01/2021 00:00","%d/%m/%Y %H:%M") + datetime.timedelta(day_nr)
    end = datetime.datetime.strptime("01/01/2021 23:00","%d/%m/%Y %H:%M") + datetime.timedelta(day_nr)
    print(start,end)
    models,effect_matrix,xtest,Yp = (
        _lear_std.evaluate_lear_in_test_dataset(path_datasets_folder=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Code_Jilles\Model_AdvForecaster', \
                                                                 path_recalibration_folder=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Code_Jilles\Model_AdvForecaster/Stacked_Bar_Plot\med', dataset='Model_AdvForecaster_dataframe', \
                                                                 calibration_window=cw, begin_test_date=start, end_test_date=end, return_coef_hour=1))
    #print(effect_matrix,xtest)
    zeros = effect_matrix[14]
    real_prices_day = real_prices.iloc[day_nr,:]
    intercepts = effect_matrix[15]
    std_prices = effect_matrix[0]
    coef_dict_day = []
    coef = models[h].coef_
    Abs_Coef_lag_prices = abs(sum(coef[:96]))
    Abs_Coef_BE_load = abs(sum(coef[96:949:12]))
    Abs_Coef_BE_wind= abs(sum(coef[97:950:12]))
    Abs_Coef_BE_solar= abs(sum(coef[98:951:12]))
    Abs_Coef_CH_price= abs(sum(coef[99:952:12]))
    Abs_Coef_FR_gen= abs(sum(coef[100:953:12]))
    Abs_Coef_DE_wind= abs(sum(coef[101:954:12]))
    Abs_Coef_DE_solar= abs(sum(coef[102:955:12]))
    Abs_Coef_Oil = abs(sum(coef[103:956:12]))
    Abs_Coef_Carbon = abs(sum(coef[104:957:12]))
    Abs_Coef_Gas= abs(sum(coef[105:958:12]))
    Abs_Coef_weather= abs(sum(coef[106:959:12]))
    Abs_Coef_FR_load= abs(sum(coef[107:960:12]))
    sum_abs_coeff = (
            Abs_Coef_lag_prices
            + Abs_Coef_BE_load
            + Abs_Coef_BE_wind
            + Abs_Coef_BE_solar
            + Abs_Coef_CH_price
            + Abs_Coef_FR_gen
            + Abs_Coef_DE_wind
            + Abs_Coef_DE_solar
            + Abs_Coef_Oil
            + Abs_Coef_Carbon
            + Abs_Coef_Gas
            + Abs_Coef_weather
            + Abs_Coef_FR_load
    )
    coef_dict_day.extend([
      {"day": day.strftime("%d-%b-%Y"),
        "Abs_Coef": "Abs. Coeff. Lagged Prices",
        "value": Abs_Coef_lag_prices},
      {"day": day.strftime("%d-%b-%Y"),
       "Abs_Coef": "Abs. Coeff. BE Load & Weather",
       "value": Abs_Coef_BE_load+Abs_Coef_weather},
      {"day": day.strftime("%d-%b-%Y"),
       "Abs_Coef": "Abs. Coeff. Wind Forecast",
       "value": Abs_Coef_DE_wind+Abs_Coef_BE_wind},
      {"day": day.strftime("%d-%b-%Y"),
       "Abs_Coef": "Abs. Coeff. Solar Forecast",
       "value": Abs_Coef_DE_solar+Abs_Coef_BE_solar},
      {"day": day.strftime("%d-%b-%Y"),
       "Abs_Coef": "Abs. Coeff. Fossil Fuels",
       "value": Abs_Coef_Oil+Abs_Coef_Carbon+Abs_Coef_Gas},
      {"day": day.strftime("%d-%b-%Y"),
       "Abs_Coef": "Abs. Coeff. CH Prices",
       "value": Abs_Coef_CH_price},
        {"day": day.strftime("%d-%b-%Y"),
         "Abs_Coef": "Abs. Coeff. FR Load & Generation",
     "value": Abs_Coef_FR_load+Abs_Coef_FR_gen}])
    return coef_dict_day,day,real_prices_day,std_prices

def generate_contributions_effect_plot(name_dataframe,calibration_window, begin_plot_date=None,end_plot_date=None):
    """

    Parameters
    ----------
    name_dataframe

    path_real_prices

    day : should have the format YYYY-mm-dd

    Returns
    -------

    """
    path_datasets_folder = str(Path.cwd().parent) + '\Datasets'
    path_forecasts_folder = str(Path.cwd().parent) + '\Contributions_for_effect_plots'
    # real_prices = pd.read_csv(path_real_prices)
    # real_prices = real_prices.set_index('Date')
    # dataframe = pd.read_csv(os.path.join(path_datasets_folder,str(name_dataframe+'.csv')))
    hourly_index = pd.date_range(start=begin_plot_date, end=end_plot_date+' 23:00', freq='H')
    data = {'datetime' : hourly_index , 'Effect_Solar' : [0] * len(hourly_index),'Effect_Wind' : [0] * len(hourly_index),
            'Effect_Lagged_Prices': [0] * len(hourly_index), 'Effect_Fossil_Fuels': [0] * len(hourly_index),
            'Effect_FR_Generation_Load': [0] * len(hourly_index),'Effect_Swiss_Prices': [0] * len(hourly_index),
            'Effect_BE_Load_Weather':[0] * len(hourly_index),'Zero_Prediction':[0] * len(hourly_index),}
    effects_dataframe = pd.DataFrame(data)
    name_csv_file = 'Effects_' + str(name_dataframe) + \
                    '_CW' + str(calibration_window) + '.csv'
    path_effects_dataframe= os.path.join(path_forecasts_folder, name_csv_file)
    for count, date in enumerate(pd.date_range(start=begin_plot_date, end=end_plot_date, freq='D')):
        models, effect_matrix, xtest, Yp = (
            _lear_std.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                path_recalibration_folder=path_forecasts_folder,
                                                dataset=str(name_dataframe), \
                                                calibration_window=calibration_window, begin_test_date=(date),
                                                end_test_date=(date) + datetime.timedelta(hours=23),
                                                return_coef_hour=1))
        #forecasted_prices = effect_matrix[0]
        zero_predictions = effect_matrix[14]
        for h in range(24):
            coef = models[h].coef_
            forecast_price_hour = effect_matrix[0][0][h]
            product = np.multiply(coef, xtest)
            s = sum(product[:])
            Effect_average = zero_predictions[0,h]
            scaler = abs(forecast_price_hour - Effect_average)/abs(s)
            # number_exog_vars = len(dataframe.columns)-2
            # number_coefficients = 96 + 72 * number_exog_vars
            # for i in range(number_exog_vars):
            effects_dataframe.iloc[count * 24 + h,1] =(sum(product[98:951:12])+sum(product[102:955:12]))* scaler #Solar
            effects_dataframe.iloc[count * 24 + h,2] =(sum(product[97:950:12])+sum(product[101:954:12]))* scaler #Wind
            effects_dataframe.iloc[count * 24 + h,3] =(sum(product[:96]))* scaler #Lagged_Prices
            effects_dataframe.iloc[count * 24 + h,4] =(sum(product[103:956:12])+sum(product[104:957:12])+sum(product[105:958:12]))* scaler #Fossil Fuels
            effects_dataframe.iloc[count * 24 + h,5] =(sum(product[100:953:12])+sum(product[107:960:12]))* scaler #FR_Generation_and_Load
            effects_dataframe.iloc[count * 24 + h,6] =(sum(product[99:952:12]))* scaler #Swiss_Prices
            effects_dataframe.iloc[count * 24 + h,7] =(sum(product[96:949:12])+sum(product[106:959:12]))* scaler #BE_Load_Weather
            effects_dataframe.iloc[count * 24 + h,8] = zero_predictions[0,h]

    effects_dataframe.to_csv(path_effects_dataframe, mode='w')



def create_effect_bar_chart(file_path_predictions,file_path_effects, day_to_plot, calibration_window,path_real_prices=None):

    forecast = pd.read_csv(file_path_predictions)
    dataframe_effects= pd.read_csv(file_path_effects)
    # real_prices = pd.read_csv(path_real_prices)
    # real_prices = real_prices.set_index('Date')
    forecast = forecast.set_index('Date')
    print(dataframe_effects)
    dataframe_effects['datetime'] = pd.to_datetime(dataframe_effects['datetime'])
    day_to_plot_dt = datetime.datetime.strptime(day_to_plot, '%Y-%m-%d')
    filtered_dataframe_dt = (dataframe_effects['datetime'] >= day_to_plot_dt) & (dataframe_effects['datetime'] <day_to_plot_dt+datetime.timedelta(days=1))
    filtered_dataframe = dataframe_effects.loc[filtered_dataframe_dt]
    real_prices_day = real_prices[filtered_dataframe_dt]
    forecasted_prices_day = forecast[filtered_dataframe_dt]
    filtered_dataframe = filtered_dataframe.drop(columns=['datetime'],axis=1)
    filtered_dataframe = pd.melt(filtered_dataframe,id_vars='Unnamed: 0', var_name='Var_Family', value_name='value')

    #Altair plotting
    #print(coef_dict_day)
    data1 = alt.Data(values=filtered_dataframe)
    bar_chart = alt.Chart(data1).mark_bar(size=12.5).encode(
            x=alt.X('hour:N', axis=alt.Axis(title='Hour of day on '+ str(day_to_plot),ticks = True)),
            y=alt.Y('sum(value):Q', axis=alt.Axis(title='Price [€/MWh]')),
            color=alt.Color('Effect:N',scale=alt.Scale(scheme='lightmulti')))
    real_and_forecasted_prices = (pd.DataFrame({'Real Prices' : real_prices_day,'Predicted Prices': forecasted_prices_day}).reset_index(drop=True))
    real_and_forecasted_prices = real_and_forecasted_prices.reset_index().rename(columns={'index':'hour'})
    data = real_and_forecasted_prices.melt('hour',var_name='type',value_name='price')
    real_and_forecasted_prices_chart = alt.Chart(data).mark_line(point=True).encode(
        x='hour:N',y='price:Q',color=alt.Color('type:N',scale=alt.Scale(range=["#276994", "#4aacc1"])))
    df2  = pd.DataFrame(forecasted_prices_day).reset_index(drop=True).reset_index()
    df2= df2.rename(columns={'index':'hour',df2.columns[1]: 'price'})
    text = alt.Chart(df2).mark_text(dy=-10,dx=-3, color='#276994').encode(
        text=alt.Text('price:Q', format='d'),
        x=alt.X('hour:N'),
        y=alt.Y('price:Q'))
    chart = alt.layer(bar_chart,real_and_forecasted_prices_chart,text).properties(title='Prices and Predictions - CW ' + str(calibration_window)+'- for ' + str(day_to_plot)).resolve_scale(color='independent')
    chart.show()

# generate_contributions_effect_plot(name_dataframe='Full_Dataset',calibration_window = 728, begin_plot_date='2020-01-01',end_plot_date='2022-12-31')

