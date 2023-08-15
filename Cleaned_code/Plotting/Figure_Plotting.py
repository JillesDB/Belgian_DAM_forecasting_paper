import matplotlib.pyplot as plt
import seaborn as sns
from Joris_plotting_original_code import evaluation
from Epftoolbox_original_code.evaluation import _gw,_ancillary_functions,_dm
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score
from pathlib import Path
plt.rcParams.update({'font.size': 14})
path_forecasts_folder = str(Path.cwd().parent) + '\Forecasts'

def scatter_plot(file_forecast,path_real_prices,name_forecast = None):
    """

    Parameters
    ----------
    file_forecast
    path_real_prices
    name_forecast

    Returns
    -------

    """
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    forecast = pd.read_csv(os.path.join(path_forecasts_folder, str(file_forecast + '.csv')))
    forecast = forecast.set_index('Date')
    real_prices_selection = real_prices.loc[forecast.index].copy()
    real_prices_selection = _ancillary_functions._transform_input_prices_for_naive_forecast(real_prices_selection, m='D', freq='1H')
    forecast = _ancillary_functions._transform_input_prices_for_naive_forecast(forecast,m='D',freq='1H')
    g = sns.scatterplot(x = real_prices_selection.iloc[:,0], y = forecast.iloc[:,0],size=6,color = "cornflowerblue", alpha= 0.6)
    z = np.polyfit(x=real_prices_selection.iloc[:,0], y= forecast.iloc[:,0],deg= 1)
    y_hat = np.poly1d(z)(real_prices.iloc[:,0])
    plt.xlabel('Actual Prices[EUR/MWh]')
    plt.ylabel(str(name_forecast)+' [EUR/MWh]')
    plt.legend(title="Scatter plot: Actual Prices and Forecast " + str(name_forecast))
    plt.show()

def line_plot(file_forecast,path_real_prices,name_forecast= None):
    """

    Parameters
    ----------
    file_forecast
    path_real_prices
    name_forecast

    Returns
    -------

    """
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    forecast = pd.read_csv(os.path.join(path_forecasts_folder,str(file_forecast+'.csv')))
    forecast = forecast.set_index('Date')
    real_prices_selection = real_prices.loc[forecast.index].copy()
    real_prices_selection = _ancillary_functions._transform_input_prices_for_naive_forecast(real_prices_selection,m='D',freq='1H')
    forecast = _ancillary_functions._transform_input_prices_for_naive_forecast(forecast,m='D',freq='1H')
    plot1 = real_prices_selection.plot(use_index = True,y = 0,linewidth = 0.4)
    plot2 = forecast.plot(y = 0,ax=plot1)
    plt.xlabel('Date')
    plt.ylabel('Prices [€/MWh]')
    plot1.legend(['Actual Prices - 2021/2022','Forecast '+str(name_forecast)])
    plt.show()

def joris_plots(path_real_prices,file_forecast,name_forecast=None):
    """

    Parameters
    ----------
    path_real_prices
    file_forecast
    name_forecast

    Returns
    -------

    """
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    forecast = pd.read_csv(os.path.join(path_forecasts_folder,str(file_forecast+'.csv')))
    forecast = forecast.set_index('Date')
    real_prices_selection = real_prices.loc[forecast.index].copy()
    real_prices_selection = _ancillary_functions._transform_input_prices_for_naive_forecast(real_prices_selection,m='D',freq='1H')
    forecast = _ancillary_functions._transform_input_prices_for_naive_forecast(forecast,m='D',freq='1H')
    eval = evaluation.Evaluation(actual=real_prices_selection.iloc[:, 0], forecast=forecast.iloc[:, 0])
    plot_J_1 = eval.plot_joint(levels=11,xlabel='Actual Prices',ylabel='Forecast ' + str(name_forecast))
    plot_J_2 = eval.plot_conditional(x_label='Actual Prices',y_label='Forecast ' + str(name_forecast),intervals=11)
    plt.show()


line_plot(path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
          name_forecast='Weighted ensemble',file_forecast='Weighted_Ensemble_LEAR_forecast_datExample_dataframe_YT0_RW1')
scatter_plot(path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
          name_forecast='Weighted ensemble',file_forecast='Weighted_Ensemble_LEAR_forecast_datExample_dataframe_YT0_RW1')
joris_plots(path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
          name_forecast='Weighted ensemble',file_forecast='Weighted_Ensemble_LEAR_forecast_datExample_dataframe_YT0_RW1')