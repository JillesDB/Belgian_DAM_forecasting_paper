"""
Classes and functions to implement the LEAR model for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
import os
import time
import sklearn.base
import scipy
from scipy.sparse import csr_matrix, find
from statsmodels.robust import mad

from sklearn.linear_model import LassoLarsIC, Lasso, LassoLars, LassoLarsCV
from Epftoolbox_original_code.data import scaling
from Epftoolbox_original_code.data import read_data
from Epftoolbox_original_code.evaluation import MAE, rMAE
import datetime

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import warnings

warnings.simplefilter("ignore", category=FutureWarning)


class LEAR(object):
    """Class to build a LEAR model, recalibrate it, and use it to predict DA electricity prices.

    An example on how to use this class is provided :ref:`here<learex2>`.

    Parameters
    ----------
    calibration_window : int, optional
        Calibration window (in days) for the LEAR model.

    """

    def __init__(self, calibration_window=364 * 3):

        # Calibration window in hours
        self.calibration_window = calibration_window

    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, Xtrain, Ytrain):
        """Function to recalibrate the LEAR model.

        It uses a training (Xtrain, Ytrain) pair for recalibration

        Parameters
        ----------
        Xtrain : numpy.array
            Input in training dataset. It should be of size *[n,m]* where *n* is the number of days
            in the training dataset and *m* the number of input features

        Ytrain : numpy.array
            Output in training dataset. It should be of size *[n,24]* where *n* is the number of days
            in the training dataset and 24 are the 24 prices of each day

        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model

        """

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Ytrain], self.scalerY = scaling([Ytrain], 'Std')

        # # Rescaling all inputs except dummies (7 last features)
        [Xtrain_no_dummies], self.scalerX = scaling([Xtrain[:, :-7]], 'Std')
        Xtrain[:, :-7] = Xtrain_no_dummies
        # Xtrain[:,:], self.scalerX =scaling([Xtrain[:, :-7]], 'Norm1')
        self.models = {}
        self.coef = {}
        time_rec_lambda = 0
        time_rec_coeff = 0
        for h in range(24):
            # Estimating lambda hyperparameter using LARS
            start_rec_lambda = time.time()
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)  # ,fit_intercept=False)#,noise_variance=0.582)
            param = param_model.fit(Xtrain, Ytrain[:, h])
            # print(param_model.intercept_)
            time_rec_lambda += (time.time() - start_rec_lambda)
            alpha = param.alpha_  # CHANGED THIS FROM alpha = param.alpha_
            #            alpha = 0.10

            # Re-calibrating LEAR using standard LASSO estimation technique
            start_rec_coeff = time.time()
            model = Lasso(max_iter=2500, alpha=alpha)  # ,fit_intercept=False)
            model.fit(Xtrain, Ytrain[:, h])
            time_rec_coeff += (time.time() - start_rec_coeff)
            # print(model.intercept_)

            self.models[h] = model
            self.coef[h] = model.sparse_coef_
        return time_rec_lambda,time_rec_coeff

    def predict(self, X, return_coef_hour=0):
        """Function that makes a prediction using some given inputs.

        Parameters
        ----------
        X : numpy.array
            Input of the model.

        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros(24)
        intercepts = np.zeros(24)
        # # Rescaling all inputs except dummies (7 last features)
        X_no_dummies = self.scalerX.transform(X[:, :-7])
        X[:, :-7] = X_no_dummies
        # X[:,:] = self.scalerX.transform(X[:,:])

        # Predicting the current date using a recalibrated LEAR
        if return_coef_hour == 1:
            effect_matrix = [0] * 16
            Yp2 = np.zeros(24)
            for i in range(16):
                x_copy = X.copy()
                #if i == 0 or i == 15:
                    #print('0')
                if i == 14:
                    x_copy[0, :] = 0
                elif i == 13:
                    x_copy[0, :96] = 0
                else:
                    x_copy[0, 95 + i::12] = 0
                    # effect lagged prices
                for h in range(24):
                    intercepts[h] = (self.models[h].intercept_)
                    Yp2[h] = self.models[h].predict(x_copy)
                effect_matrix[i] = self.scalerY.inverse_transform(Yp2.reshape(1, -1))
                effect_matrix[15] = intercepts
            # effect_matrix = (effect_matrix)
            return effect_matrix, X
        else:
            for h in range(24):
                Yp[h] = self.models[h].predict(X)
            Yp = self.scalerY.inverse_transform(Yp.reshape(1, -1))
            return Yp
            # Predicting test dataset and saving
            # self.predict is basically a dot product of coefficients and X (=asihn transformed Xtest data)

    def recalibrate_predict(self, Xtrain, Ytrain, Xtest, i, return_coef_hour=0,timing=0):
        """Function that first recalibrates the LEAR model and then makes a prediction.

        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.

        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset.
        Xtest : numpy.array
            Input of the test dataset.
        Ytrain : numpy.array
            Output of the training dataset.

        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """
        time_pred=0
        if i == 0:
            time_rec_lambda, time_rec_coeff = self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
        if len(Xtest) == 0:
            print('xtest is empty')
            return
        if return_coef_hour == 1:
            # self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
            effect_matrix, xtest = self.predict(X=Xtest, return_coef_hour=return_coef_hour)
            return effect_matrix, xtest
        else:
            start_pred = time.time()
            Yp = self.predict(X=Xtest, return_coef_hour=return_coef_hour)
            time_pred = time.time() - start_pred
            return Yp, [time_rec_lambda,time_rec_coeff,time_pred]

    def _build_and_split_XYs(self, df_train, df_test=None, date_test=None):

        """Internal function that generates the X,Y arrays for training and testing based on pandas dataframes

        Parameters
        ----------
        df_train : pandas.DataFrame
            Pandas dataframe containing the training data

        df_test : pandas.DataFrame
            Pandas dataframe containing the test data

        date_test : datetime, optional
            If given, then the test dataset is only built for that date

        Returns
        -------
        list
            [Xtrain, Ytrain, Xtest] as the list containing the (X,Y) input/output pairs for training,
            and the input for testing
        """

        # Checking that the first index in the dataframes corresponds with the hour 00:00
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        #
        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        # 96 prices + n_exogenous * (24 * 3 day lags) + 7 weekday dummies
        # Price lags: D-1, D-2, D-3, D-7
        # Exogeneous inputs lags: D, D-1, D-7
        n_features = 96 + n_exogenous_inputs * 72 + 7

        # Extracting the predicted dates for testing and training. We leave the first week of data
        # out of the prediction as the maximum lag can be one week

        # We define the potential time indexes that have to be forecasted in training
        # and testing
        indexTrain = df_train.loc[df_train.index[0] + pd.Timedelta(days=7):].index

        # For testing, the test dataset is different whether depending on whether a specific test
        # dataset is provided
        if date_test is None:
            indexTest = df_test.loc[df_test.index[0] + pd.Timedelta(days=7):].index
        else:
            indexTest = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        # We extract the prediction dates/days.
        # predDatesTrain = indexTrain.round('1H')[::24]
        # predDatesTest = indexTest.round('1H')[::24]
        predDatesTrain = indexTrain.round('1H')[::24]
        predDatesTest = indexTest.round('1H')[::24]
        # We create two dataframe to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day
        indexTrain = pd.DataFrame(index=predDatesTrain, columns=['h' + str(hour) for hour in range(24)])
        indexTest = pd.DataFrame(index=predDatesTest, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
            indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)
            # indexTrain[indexTrain.columns['h' + str(hour)]] = indexTrain.index + pd.Timedelta(hours=hour)
            # indexTest[indexTest.columns['h' + str(hour)]] = indexTest.index + pd.Timedelta(hours=hour)

        # Preallocating in memory the X and Y arrays
        Xtrain = np.zeros([indexTrain.shape[0], n_features])  # added +1
        Xtest = np.zeros([indexTest.shape[0], n_features])
        Ytrain = np.zeros([indexTrain.shape[0], 24])

        # Index that
        feature_index = 0

        #
        # Adding the historial prices during days D-1, D-2, D-3, and D-7
        # For each hour of a day
        for hour in range(24):
            # For each possible past day where prices can be included
            for past_day in [1, 2, 3, 7]:
                # We define the corresponding past time indexes using the auxiliary dataframes
                pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                                 pd.Timedelta(hours=24 * past_day)
                pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                                pd.Timedelta(hours=24 * past_day)

                # We include the historical prices at day D-past_day and hour "h"
                Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Price']
                Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Price']
                feature_index += 1

        #
        # Adding the exogenous inputs during days D, D-1,  D-7
        #
        # For each hour of a day
        for hour in range(24):
            # For each possible past day where exogenous inputs can be included
            for past_day in [1, 7]:  # changed from for past_Day in [1,2]
                # For each of the exogenous input
                for exog in range(1, n_exogenous_inputs + 1):
                    # Definying the corresponding past time indexs using the auxiliary dataframses
                    pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                                     pd.Timedelta(hours=24 * past_day)
                    pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                                    pd.Timedelta(hours=24 * past_day)

                    # Including the exogenous input at day D-past_day and hour "h"
                    Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Exogenous ' + str(exog)]
                    Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Exogenous ' + str(exog)]
                    feature_index += 1

            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exogenous_inputs + 1):
                # Definying the corresponding future time indexs using the auxiliary dataframes
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

                # Including the exogenous input at day D and hour "h"
                Xtrain[:, feature_index] = df_train.loc[futureIndexTrain, 'Exogenous ' + str(exog)]
                Xtest[:, feature_index] = df_test.loc[futureIndexTest, 'Exogenous ' + str(exog)]
                feature_index += 1

        #
        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        #
        # For each day of the week
        for dayofweek in range(7):
            Xtrain[indexTrain.index.dayofweek == dayofweek, feature_index] = 1
            Xtest[indexTest.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1

        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
            futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

            # Extracting Y value based on time indexs
            Ytrain[:, hour] = df_train.loc[futureIndexTrain, 'Price']
        self.Xtrain = Xtrain
        return Xtrain, Ytrain, Xtest

    def recalibrate_and_forecast_next_day(self, df, calibration_window, next_day_date, begin_test_date,
                                          recalibration_window=1, i=0, return_coef_hour=0,timing = 0):
        begin_test_date_dt = pd.to_datetime(begin_test_date)
        # We define the new training dataset and test datasets
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        df_train0 = df.loc[:begin_test_date_dt - pd.Timedelta(hours=1)]

        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]
        df_train0 = df_train0.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features.
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]
        # next_day_date + pd.Timedelta(hours = (recalibration_window-1)*24):
        # Generating X,Y pairs for predicting prices
        if recalibration_window == 0:
            Xtrain, Ytrain, Xtest, = self._build_and_split_XYs(
                df_train=df_train0, df_test=df_test, date_test=next_day_date)
        else:
            Xtrain, Ytrain, Xtest = self._build_and_split_XYs(
                df_train=df_train, df_test=df_test, date_test=next_day_date)
            # .iloc[:-(recalibration_window- 1) * 24, :]
        # Recalibrate the LEAR model and extract the prediction.
        if return_coef_hour == 1:
            effect_matrix, xtest = self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, i=i,
                                                            return_coef_hour=return_coef_hour)
            return effect_matrix, xtest
        else:
            Yp, [time_rec_lambda,time_rec_coeff,time_pred] = self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, i=i,
                                          return_coef_hour=return_coef_hour,timing=timing)
            return Yp, [time_rec_lambda,time_rec_coeff,time_pred]


def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('../../../../epftoolbox/models', 'datasets'),
                                  path_recalibration_folder=os.path.join('../../../../epftoolbox/models',
                                                                         'experimental_files'),
                                  dataset='PJM', years_test=2, calibration_window=364 * 3,
                                  begin_test_date=None, end_test_date=None, recalibration_window=1, return_coef_hour=0,
                                  timing=0):
    start = time.time()
    # Checking if provided directory for recalibration exists and if not create it
    if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)
    begin_test_date_td = pd.to_datetime(begin_test_date)
    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)

    # Defining unique name to save the forecast
    forecast_file_name = 'LEAR_forecast' + '_dataframe_' + str(dataset) + \
                         '_CW' + str(calibration_window) + '_RW' + str(recalibration_window) + '.csv'

    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)
    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])

    real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

    forecast_dates = forecast.index

    model = LEAR(calibration_window=calibration_window)
    # For loop over the recalibration dates
    # The forecaster each time makes a forecast for all dates in one RW
    # if the forecast has already been made, it shouldn't be made again
    # and so the forecaster should move on until having reached the last date.
    timing_dataframe = pd.DataFrame(0, index=forecast_dates, columns=['time_rec_lambda ' + str(calibration_window),
                                                                      'time_rec_coeff ' + str(calibration_window),
                                                                      'time_pred ' + str(calibration_window)])
    for date in forecast_dates:
        if pd.isna(forecast.loc[date, 'h3']):
            for i in range(recalibration_window):

                # For simulation purposes, we assume that the available data is
                # the data up to current date where the prices of current date are not known
                current_date = date + pd.Timedelta(hours=(i) * 24)
                # current date shouldn't run further than the end_test_date, so then the code should finish.
                if current_date > pd.to_datetime(end_test_date):
                    return forecast
                if date + pd.Timedelta(hours=23) + pd.Timedelta(hours=(recalibration_window - 1) * 24) < pd.to_datetime(
                        end_test_date):
                    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23) + pd.Timedelta(
                        hours=(recalibration_window - 1) * 24), :]], axis=0)
                else:  # (date + pd.Timedelta(hours=23) + pd.Timedelta(hours=(recalibration_window - 1) * 24) == pd.to_datetime(end_test_date):
                    data_available = pd.concat([df_train, df_test.loc[:end_test_date, :]], axis=0)

                # We set the real prices for current date to NaN in the dataframe of available data
                data_copy = data_available
                data_copy.loc[current_date:current_date + pd.Timedelta(hours=23), 'Price'] = np.NaN

                # Recalibrating the model with the most up-to-date available data and making a prediction
                # for the next day
                Yp,[time_rec_lambda,time_rec_coeff,time_pred] = model.recalibrate_and_forecast_next_day(df=data_copy,
                                                             next_day_date=date + pd.Timedelta(hours=i * 24),
                                                             calibration_window=calibration_window,
                                                             recalibration_window=recalibration_window,
                                                             begin_test_date=begin_test_date_td, i=i,timing=timing)
                if return_coef_hour == 1:
                    effect_matrix, xtest = model.recalibrate_and_forecast_next_day(df=data_copy,
                                                                                   next_day_date=date + pd.Timedelta(
                                                                                       hours=i * 24),
                                                                                   calibration_window=calibration_window,
                                                                                   recalibration_window=recalibration_window,
                                                                                   begin_test_date=begin_test_date_td,
                                                                                   i=i,
                                                                                   return_coef_hour=return_coef_hour)
                    models = model.models
                    xtest = np.squeeze(np.asarray(xtest))
                    return models, effect_matrix, xtest, Yp
                # Saving the current prediction
                forecast.loc[date + pd.Timedelta(hours=i * 24), :] = Yp
                # Saving forecast
                forecast.to_csv(forecast_file_path)
                if timing:
                    timing_dataframe.loc[date] = time_rec_lambda,time_rec_coeff,time_pred
                    #timing_dataframe.loc[date,1] =time_pred

        else:
            continue
    if timing:
        return forecast,timing_dataframe
    else:
        return forecast

# evaluate_lear_in_test_dataset(path_datasets_folder='.',path_recalibration_folder='.',dataset='BE',years_test=1,
#                           calibration_window=56,begin_test_date=None,end_test_date=None)