#  MIT License
#
#  Copyright (c) 2019 Oleksii Lialka
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')


class HoltWinters:
    """
    Holt-Winters model with the anomalies detection using Brutlag method

    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)

    """

    def __init__(self, series, slen, alpha, beta, gamma,
                 n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(
                self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[
                    self.slen * j:self.slen * j + self.slen]) / float(
                    self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - \
                                        season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0]
                                      + self.scaling_factor
                                      * self.PredictedDeviation[0])

                self.LowerBond.append(self.result[0]
                                      - self.scaling_factor
                                      * self.PredictedDeviation[0])
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append(
                    (smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(
                    self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (
                        val - seasonals[i % self.slen]) + (
                                              1 - self.alpha) * (
                                              smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (
                        1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * \
                                           (val - smooth) + (1 - self.gamma) * \
                                           seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] +
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] -
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)

            self.Season.append(seasonals[i % self.slen])

    def plot(self, plot_intervals=False, plot_anomalies=False):
        """
            series - dataset with timeseries
            plot_intervals - show confidence intervals
            plot_anomalies - show anomalies
        """

        plt.figure(figsize=(10, 5))
        plt.plot(self.result, label="Model")
        plt.plot(self.series.values, label="Actual")
        error = mean_absolute_percentage_error(self.series.values,
                                               self.result[:len(self.series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(self.series))
            anomalies[self.series.values < self.LowerBond[:len(self.series)]] = \
                self.series.values[
                    self.series.values < self.LowerBond[:len(self.series)]]
            anomalies[self.series.values > self.UpperBond[:len(self.series)]] = \
                self.series.values[
                    self.series.values > self.UpperBond[:len(self.series)]]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(self.UpperBond, "r--", alpha=0.5,
                     label="Up/Low confidence")
            plt.plot(self.LowerBond, "r--", alpha=0.5)
            plt.fill_between(x=range(0, len(self.result)), y1=self.UpperBond,
                             y2=self.LowerBond, alpha=0.2, color="grey")

        plt.vlines(len(self.series), ymin=min(self.LowerBond),
                   ymax=max(self.UpperBond), linestyles='dashed')
        plt.axvspan(len(self.series) - 20, len(self.result), alpha=0.3,
                    color='lightgrey')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best", fontsize=13)
        plt.show()

    def plotBrutlags(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.PredictedDeviation)
        plt.grid(True)
        plt.axis('tight')
        plt.title("Brutlag's predicted deviation")
        plt.show()


def cross_validation_score(params, series,
                           loss_function=mean_squared_error, slen=60):
    """
    Estimate error of cross-validation
    :param params: parameters of the model
    :param series: time series
    :param loss_function: objective function of errors
    :param slen: length of the season
    :return:
    """
    # errors array
    errors = []

    values = series.values
    alpha, beta, gamma = params

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train],
                            slen=slen,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            n_preds=len(test)
                            )

        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
    return np.mean(np.array(errors))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    # import data
    series = pd.read_csv('../data/^GSPC.csv', index_col=['Date'], skiprows=range(1, 1000), usecols=['Date', 'Open'],
                         parse_dates=['Date'])

    # Initialize parameters of optimization problem
    season = 60  # 60-day seasonality
    data = series.Open[:-season]
    mean_abs_percentage_error = mean_absolute_percentage_error
    parameters = data, mean_abs_percentage_error, season

    constraints = (0, 1), (0, 1), (0, 1)
    x = [0, 0, 0]  # Initial guess

    # Objective function to be minimized
    objective_function = cross_validation_score

    # Perform cross-validation of Holt-Winters model
    # optimization algorithms: TNC, CG
    opt = minimize(objective_function, x0=x,
                   args=parameters,
                   method='CG', bounds=constraints)

    # Retrieve optimized parameters
    alpha_opt, beta_opt, gamma_opt = opt.x
    print('Optimal parameters for Holt-Winters model:'
          '\nalpha = {} \nbeta = {} \ngamma = {}\n'.format(alpha_opt, beta_opt, gamma_opt))

    # Initialize optimized model
    model = HoltWinters(data, slen=season,
                        alpha=alpha_opt,
                        beta=beta_opt,
                        gamma=gamma_opt,
                        n_preds=60, scaling_factor=3)

    # Run time series approximation
    model.triple_exponential_smoothing()

    # Plot results of the model against actual values
    model.plot(plot_intervals=True, plot_anomalies=True)

    # Display deviation of results
    model.plotBrutlags()


if __name__ == '__main__':
    main()
