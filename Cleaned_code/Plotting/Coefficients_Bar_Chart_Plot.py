import pandas as pd
from Epftoolbox_original_code import _lear
import datetime
from pathlib import Path
import os
import altair as alt
alt.data_transformers.disable_max_rows()

import numpy as np

cwd = Path.cwd().parent
path_datasets_folder = os.path.join(cwd,'Datasets')
path_coeff_folder = os.path.join(cwd,'Dataframes_with_Coefficients')
path_real_prices = pd.read_csv(os.path.join(path_datasets_folder,'Real_prices.csv'))

def create_coef_analysis_dict(day_to_plot,calibration_window,name_dataframe):
    path_datasets_folder = str(Path.cwd().parent) + '\Datasets'
    path_forecasts_folder = str(Path.cwd().parent) + '\Forecasts_for_plots'
    dataframe = pd.read_csv(os.path.join(path_datasets_folder,str(name_dataframe+'.csv')))
    models, effect_matrix, xtest, Yp = (_lear.evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder, \
                                                                 path_recalibration_folder= path_forecasts_folder, dataset=str(name_dataframe), \
                                                                 calibration_window=calibration_window, begin_test_date= str(day_to_plot) + ' 00:00',
                                                                            end_test_date=str(day_to_plot) + ' 23:00', return_coef_hour=1))
    coef_dict_day = []
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

    return coef_dict_day

def create_coefficient_bar_chart(name_dataframe, path_real_prices, day_to_plot,calibration_window,file_path=None):
    """

    Parameters
    ----------
    name_dataframe

    path_real_prices

    day : should have the format YYYY-mm-dd

    Returns
    -------

    """

    if file_path is None:
        coef = create_coef_analysis_dict(day_to_plot,calibration_window=calibration_window)
        filtered_dataframe = alt.Data(values=coef)
    else:

        dataframe = pd.read_csv(file_path)
        print(dataframe)
        dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
        #dataframe = dataframe.set_index('datetime')
        #dataframe = dataframe.drop(columns=['Unnamed: 0'],axis=1)
        day_to_plot_dt = datetime.datetime.strptime(day_to_plot, '%Y-%m-%d')
        #print(dataframe.index.date,day_to_plot_dt)
        filtered_dataframe = (dataframe['datetime'] >= day_to_plot_dt) & (dataframe['datetime'] <day_to_plot_dt+datetime.timedelta(days=1))
        filtered_dataframe = dataframe.loc[filtered_dataframe]
        #dataframe = dataframe[dataframe.index.date == day_to_plot_dt]
        #filtered_dataframe.index = dataframe.index.hour
        #dataframe = dataframe.reset_index(drop = False)
        filtered_dataframe = filtered_dataframe.drop(columns=['datetime'],axis=1)
        #filtered_dataframe.index.name = 'hour'
        print(filtered_dataframe)
        filtered_dataframe = pd.melt(filtered_dataframe,id_vars='Unnamed: 0', var_name='Var_Family', value_name='value')
        print(filtered_dataframe)
    #Altair plotting
    #data1 = alt.Data(values=filtered_dataframe)
    bar_chart = alt.Chart(filtered_dataframe).mark_bar(size=12.5).encode(
        x=alt.X('Unnamed: 0:N', axis=alt.Axis(title='Hour of day on ' + str(day_to_plot), ticks=True)),
        y=alt.Y('value:Q', axis=alt.Axis(title='Absolute Value Coefficients'), stack="zero"),
        color=alt.Color('Var_Family:N',scale=alt.Scale(range=["#c7ead4", "#b4e0aa", "#c5e08b", "#e5e079", "#f6d264", "#f5b34c", "#f4913e"]))#, scale=alt.Scale(scheme='lightmulti'))
    )
    text = alt.Chart(filtered_dataframe).mark_text(dy=0.5, color='black', baseline='middle', fontSize=6).encode(
        # ,baseline='line-top'
        x=alt.X('Unnamed: 0:N'),
        y=alt.Y('sum(value):Q', stack='zero')
        , detail='Var_Family:N',
        text=alt.Text('sum(value):Q', format='.2f')
    ).transform_filter(
        {'or': [alt.FieldGTPredicate(field='value', gt=0.04), alt.FieldLTPredicate(field='value', lt=-0.04)]})
    # combine the bar chart, text and point layers
    chart = alt.layer(bar_chart,text).properties(title='Absolute Coefficients - CW' + str(calibration_window) + ' '+ str(day_to_plot)).resolve_scale(
        color='independent')
    # display the chart
    chart.show()

# create_coefficient_bar_chart(name_dataframe='Example_dataframe', path_real_prices=path_real_prices,
#                          day_to_plot='2020-01-01',calibration_window=56,file_path=os.path.join(path_coeff_folder,'Data_clock_plot_dataframe_Example_dataframe_CW56.csv'))
