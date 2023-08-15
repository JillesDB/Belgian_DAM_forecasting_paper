import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pylab as plt
import datetime
import pulp
import os
from Epftoolbox_original_code.evaluation import _ancillary_functions
from pathlib import Path
path_forecasts_folder = str(Path.cwd().parent) +  '\Forecasts'
print(path_forecasts_folder)
plt.rcParams.update({'font.size': 12})

def plot_day(solution,day,predicted_prices_day,real_prices_day):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(solution, '*-', label='Pumping/Generation Capacity [MW]',color='limegreen',linewidth = 4)
    ax2.plot(real_prices_day, 'o--', label='Real Price [€/MWh]',color='royalblue',linewidth = 4)
    ax2.plot(predicted_prices_day, 'o--', label='Predicted Price [€/MWh]',color='yellow',linewidth = 4)
    ax1.grid(True)
    ax1.set_xlabel('Time [h]')
    plt.title('Pumping/Generating on '+str(day))
    ax1.set_ylabel('Pumping/Generation Capacity [MW]', color='black')
    ax2.set_ylabel('Real and Predicted Prices [€/MWh]', color='black')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2 , labels + labels2, loc=6)
    plt.show()


    plt.scatter(real_prices_day, solution,s=100)
    plt.xlabel('Real Prices [€/MWh]')
    plt.ylabel('Pumping/Generation Capacity [MW]')
    plt.grid(True)
    plt.title('Operation Modes and Real Prices on '+str(day))
    plt.show()


def maxProfit(predicted_prices,storage_capacity,generator_capacity):
    time_points = [str(i) for i in range(len(predicted_prices))]
    forecast_price_dict = dict(zip(time_points, predicted_prices))
    Pcharge = [pulp.LpVariable('Pcharge_{}'.format(i), lowBound=0, upBound=generator_capacity) for i in time_points]
    Pdischarge = [pulp.LpVariable('Pdischarge_{}'.format(i), lowBound=0, upBound=generator_capacity) for i in time_points]
    time_point_vars = pulp.LpVariable.dict("solution", time_points, -generator_capacity, generator_capacity)
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    prob += pulp.lpSum([(time_point_vars[i]) * forecast_price_dict[i] for i in time_points])
    for j in range(len(time_points)):
        # At every point in time the cumulated charge/discharge needs to be within bounds
        prob += pulp.lpSum([time_point_vars[i] for i in time_points[:j + 1]]) <= storage_capacity
        prob += pulp.lpSum([time_point_vars[i] for i in time_points[:j + 1]]) >= 0
    prob.solve()
    solution_dict = {}
    for v in prob.variables():
        solution_dict[int(v.name.replace("solution_", ""))] = v.varValue
    # Now convert solution dictionary to a time-sorted list
    solution = []
    for i in sorted(solution_dict):
        solution.append(solution_dict[i])
    return solution

def profit_calculator(name_forecast,path_real_prices,generator_capacity = 1080,storage_capacity = 6*1080,dates_to_plot=None):
    """

    Parameters
    ----------
    name_forecast: the name of the csv file in the Forecasts folder with the price forecast.
    it should be size m x 24 with m the index as dates (format YYYY-mm-dd) and the 24 columns (h0,h1,...,h23)
    with the forecast price for each hour.

    path_real_prices: full path of the actual prices, with a similar format as the forecast prices.

    generator_capacity : the capacity of the generation/consumption of electricity of the flexible storage unit.
    Assuming the prices are in EUR/MWh, the generator capacity should have the unit MW

    storage_capacity: the max energy the flexible storage unit can store. Again, assuming prices have the unit
    EUR/MWh, then the storage_capacity should have unit MWh

    dates_to_plot: a set with dates that you want to have plotted, formatted as: {'YYYY-mm-dd'}

    Returns
    A list containing the monetary value of the flexible storage unit when deploying the unit using your forecast
    of the prices, a weekly naive forecast of the prices and the maximal profit that can be obtained.
    Furthermore, the function will generate plots visualizing the flexible unit's deployment for the dates in the
    set dates_to_plot
    -------

    """
    forecast_profit = 0
    max_profit = 0
    naive_profit = 0
    real_prices = pd.read_csv(path_real_prices)
    real_prices = real_prices.set_index('Date')
    path_file = os.path.join(path_forecasts_folder, name_forecast)
    forecast = pd.read_csv(path_file)
    forecast = forecast.set_index('Date')
    real_prices_selection = real_prices.loc[forecast.index].copy()
    naive_prices = real_prices.shift(7,axis = 0)
    naive_prices_selection = naive_prices.loc[forecast.index].copy()
    for i in forecast.index:
        print(i)
        real_prices_day = real_prices_selection.loc[i].tolist()
        forecasted_prices_day = forecast.loc[i].tolist()
        naive_day_prices = naive_prices_selection.loc[i].tolist()
        a = maxProfit(forecasted_prices_day, storage_capacity, generator_capacity)
        b = maxProfit(naive_day_prices, storage_capacity, generator_capacity)
        c = maxProfit(real_prices_day, storage_capacity, generator_capacity)
        forecast_profit += sum([x * y for x, y in zip(a, real_prices_day)])
        naive_profit += sum([x * y for x, y in zip(b, real_prices_day)])
        max_profit += sum([x * y for x, y in zip(c, real_prices_day)])
        if i in dates_to_plot:
            plot_day(a, day=i,real_prices_day=real_prices_day,predicted_prices_day=forecasted_prices_day)
    return [-forecast_profit,-max_profit,-naive_profit]

dates =  {'2021-01-07','2021-01-08'}

profits = profit_calculator(name_forecast='Ensemble_LEAR_forecast_datExample_dataframe_YT0_RW1.csv',
                  path_real_prices=r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Cleaned_code\Datasets\Real_prices.csv',
                  dates_to_plot=dates)
print(profits)