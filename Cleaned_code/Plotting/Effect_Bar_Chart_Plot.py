import pandas as pd
from Epftoolbox_original_code import _lear
import datetime
from pathlib import Path
import os
import altair as alt
import numpy as np

def plot_effect_bar_chart(name_dataframe, path_real_prices, day_to_plot,calibration_window):
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
    path_forecasts_folder = str(Path.cwd().parent) + '\Forecasts_for_plots'
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    real_prices_day = real_prices.loc[day_to_plot]
    dataframe = pd.read_csv(os.path.join(path_datasets_folder,str(name_dataframe+'.csv')))
    models, effect_matrix, xtest, Yp = (_lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                                 path_recalibration_folder= path_forecasts_folder, dataset=str(name_dataframe), \
                                                                 calibration_window=calibration_window, begin_test_date= str(day_to_plot) + ' 00:00',
                                                                            end_test_date=str(day_to_plot) + ' 23:00', return_coef_hour=1))
    coef_dict_day = []
    forecasted_prices = effect_matrix[0]
    zeros = effect_matrix[14]
    for h in range(24):
        coef = models[h].coef_
        forecast_price_hour = effect_matrix[0][0][h]
        product = np.multiply(coef, xtest)
        s = sum(product[:])
        Effect_average = zeros[0,h]
        scaler = abs(forecast_price_hour - Effect_average)/abs(s)
        number_exog_vars = len(dataframe.columns)-2
        number_coefficients = 96 + 72 * number_exog_vars
        coef_dict_day.extend([
            {"hour": h,
             "Effect": "Zero Prediction",
             "value": Effect_average},
            {"hour": h,
             "Effect": "Effect Lagged Prices",
             "value": sum(product[:96])*scaler}])
        for i in range(number_exog_vars):
            coef_dict_day.extend([{"hour": h,
                                   "Effect": str(dataframe.columns.tolist()[i+2]),
                                   "value": sum(product[96+i:(number_coefficients-number_exog_vars+i+1):number_exog_vars])*scaler}])

    #Altair plotting
    print(coef_dict_day)
    data1 = alt.Data(values=coef_dict_day)
    bar_chart = alt.Chart(data1).mark_bar(size=12.5).encode(
            x=alt.X('hour:N', axis=alt.Axis(title='Hour of day on '+ str(day_to_plot),ticks = True)),
            y=alt.Y('sum(value):Q', axis=alt.Axis(title='Price [€/MWh]')),
            color=alt.Color('Effect:N',scale=alt.Scale(scheme='lightmulti')))
    real_and_forecasted_prices = (pd.DataFrame({'Real Prices' : real_prices_day,'Predicted Prices': forecasted_prices[0]}).reset_index(drop=True))
    real_and_forecasted_prices = real_and_forecasted_prices.reset_index().rename(columns={'index':'hour'})
    data = real_and_forecasted_prices.melt('hour',var_name='type',value_name='price')
    real_and_forecasted_prices_chart = alt.Chart(data).mark_line(point=True).encode(
        x='hour:N',y='price:Q',color=alt.Color('type:N',scale=alt.Scale(range=["#276994", "#4aacc1"])))
    df2  = pd.DataFrame(forecasted_prices[0]).reset_index(drop=True).reset_index()
    df2= df2.rename(columns={'index':'hour',df2.columns[1]: 'price'})
    text = alt.Chart(df2).mark_text(dy=-10,dx=-3, color='#276994').encode(
        text=alt.Text('price:Q', format='d'),
        x=alt.X('hour:N'),
        y=alt.Y('price:Q'))
    chart = alt.layer(bar_chart,real_and_forecasted_prices_chart,text).properties(title='Prices and Predictions - CW ' + str(calibration_window)+'- for ' + str(day_to_plot)).resolve_scale(color='independent')
    chart.show()
plot_effect_bar_chart(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
                         day_to_plot='2021-01-01',calibration_window=56)