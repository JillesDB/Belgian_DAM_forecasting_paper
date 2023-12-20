import pandas as pd
from evaluation import Evaluation
import numpy as np
import scipy.stats as stat
from matplotlib import pyplot as plt

dataframe_prices = pd.read_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Code_Jilles\Error_analysis\scatter_plot.csv')
dataframe_exog = pd.read_csv(r'C:\Users\r0763895\Documents\Masterthesis\Masterthesis\Code\epftoolbox\Code_Jilles\Error_analysis\Dataframe.csv')
dataframe_prices = dataframe_prices.set_index('Date')
dataframe_exog = dataframe_exog.set_index('Date')
eval = Evaluation(actual=dataframe_prices['real prices'], forecast=dataframe_prices['My Predictions'])
#eval.metrics()
plot1 = eval.plot_joint(levels=5)
plot2 = eval.plot_conditional(x_label='real prices',y_label='Predictions',intervals=11)
plt.show()

