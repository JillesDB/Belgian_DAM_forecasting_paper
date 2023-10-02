import pandas as pd
from Epftoolbox_original_code import _lear
import datetime
from pathlib import Path
import os
import altair as alt

def plot_coefficient_bar_chart(name_dataframe, path_real_prices, day_to_plot,calibration_window,file_path=None):
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
    forecast_prices_day = effect_matrix[0]
    for h in range(24):
        coef = models[h].coef_
        number_exog_vars = len(dataframe.columns)-2
        number_coefficients = 96 + 72 * number_exog_vars
        Abs_Coef_lag_prices = abs(sum(coef[:96]))
        coef_dict_day.extend([
            {"hour": h,
             "Abs_Coef": "Abs. Coeff. Lagged Prices",
             "value": Abs_Coef_lag_prices}])
        for i in range(number_exog_vars):
            coef_dict_day.extend([{"hour": h,
                                   "Abs_Coef": str(dataframe.columns.tolist()[i+2]),
                                   "value": abs(sum(coef[(96+i):(number_coefficients-number_exog_vars+i+1):number_exog_vars]))}])

    dataframe = pd.read_csv(file_path)
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe.drop(columns=['Unnamed: 0'],axis=1)
    dataframe = dataframe[dataframe.index.date == day_to_plot]
    dataframe.index = dataframe.index.dt.hour
    dataframe = dataframe.reset_index(drop = False)
    dataframe.index.name = 'hour'
    dataframe.melt('hour', var_name='Var_Family', value_name='value')

    #Altair plotting
    data1 = alt.Data(values=coef_dict_day)
    bar_chart = alt.Chart(data1).mark_bar(size=12.5).encode(
        x=alt.X('hour:N', axis=alt.Axis(title='Hour of day on ' + str(day_to_plot), ticks=True)),
        y=alt.Y('sum(value):Q', axis=alt.Axis(title='Absolute Value Coefficients'), stack="zero"),
        color=alt.Color('Abs_Coef:N', scale=alt.Scale(scheme='lightmulti'))
    )
    text = alt.Chart(data1).mark_text(dy=0.5, color='black', baseline='middle', fontSize=6).encode(
        # ,baseline='line-top'
        x=alt.X('hour:N'),
        y=alt.Y('sum(value):Q', stack='zero')
        , detail='Abs_Coef:N',
        text=alt.Text('sum(value):Q', format='.2f')
    ).transform_filter(
        {'or': [alt.FieldGTPredicate(field='value', gt=0.04), alt.FieldLTPredicate(field='value', lt=-0.04)]})
    # combine the bar chart, text and point layers
    chart = alt.layer(bar_chart).properties(title='Absolute Coefficients - CW' + str(calibration_window) + ' '+ str(day_to_plot)).resolve_scale(
        color='independent')
    # display the chart
    chart.show()

plot_coefficient_bar_chart(name_dataframe='Example_dataframe', path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
                         day_to_plot='2021-01-01',calibration_window=56)