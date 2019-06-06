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
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')


class MovingAverage:
    """
    Master class for Moving Average, Exponential MA, Double-Exp-MA
    """

    def __init__(self, series):
        """
        :param series: pandas dataframe with time series
        """
        self.series = series
        self.window = None
        self.SimpleMovingAverage = None

    def simple(self, window):
        """
        Constructs MA(n) of time series with specified window
        :param window: size of the rolling window
        :return: datapoints of simple MA
        """
        self.window = window
        self.SimpleMovingAverage = self.series.rolling(window=window).mean()
        return self.SimpleMovingAverage

    def plot_simple(self, std_scale=2, intervals=False, anomalies=False):
        """
        Plot simple MA against actual data
        :param std_scale: multiplier of standard deviation
        :param intervals: plot confidence intervals
        :param anomalies: plot abnormal values
        :return: None
        """
        line = self.SimpleMovingAverage

        plt.figure(figsize=(10, 5))
        plt.title("Moving average({})".format(self.window))
        plt.plot(line, "g", label="Rolling mean trend")

        if intervals:
            mae = mean_absolute_error(self.series[self.window:],
                                      line[self.window:])
            std = np.std(self.series[self.window:] - line[self.window:])
            lower_bond = line - (mae + std_scale * std)
            upper_bond = line + (mae + std_scale * std)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            if anomalies:
                anomalies = pd.DataFrame(index=self.series.index,
                                         columns=self.series.columns)
                anomalies[self.series < lower_bond] = self.series[
                    self.series < lower_bond]
                anomalies[self.series > upper_bond] = self.series[
                    self.series > upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()


class ExponentialMovingAverage(MovingAverage):

    def __init__(self, series):
        super().__init__(series)
        self.alpha = None
        self.beta = None
        self.gamma = None

    def basic_smoothing(self, alpha):
        """
        Apply simple exponential smoothing to time series
        :param alpha: smoothing parameter fof level
        :return: smoothed datapoints
        """
        self.alpha = alpha
        result = [self.series['Adj Close'][0]]  # first value is same as series
        for n in range(1, len(self.series)):
            result.append(
                self.alpha * self.series['Adj Close'][n] + (1 - self.alpha) *
                result[
                    n - 1])
        self.basic_ema = result
        return result

    def plot_basic(self):
        """
        Plot exponential smoothing against actual data
        """
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 5))

            plt.plot(self.basic_ema, label="Alpha {}".format(self.alpha))

            plt.plot(self.series.values, "c", label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing")
            plt.grid(True)
            plt.show()

    def double_smoothing(self, alpha, beta):
        """
        Apply double exponential smoothing to time series
        :param alpha: smoothing parameter for level
        :param beta: smoothing parameter for trend
        :return: smoothed datapoints
        """
        self.alpha = alpha
        self.beta = beta
        # first value is same as series
        data = self.series['Adj Close']
        result = [data[0]]
        for n in range(1, len(data) + 1):
            if n == 1:
                level, trend = data[0], data[1] - data[0]
            if n >= len(data):  # forecasting
                value = result[-1]
            else:
                value = data[n]
            last_level, level = level, self.alpha * value + (
                    1 - self.alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        self.double_ema = result
        return result

    def plot_double(self):
        """
        Plot double exponential smoothing against actual data
        """
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 5))

            plt.plot(self.double_ema,
                     label="Alpha {}, beta {}".format(self.alpha, self.beta))
            plt.plot(self.series.values, label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()


def main():
    # import data
    path = '../data/^GSPC.csv'
    data = pd.read_csv(path,
                       index_col=['Date'],
                       usecols=['Date', 'Adj Close'],
                       skiprows=range(1, 2000),
                       parse_dates=['Date']
                       )
    # Simple MA
    sma = MovingAverage(data)
    sma.simple(window=20)
    sma.plot_simple(intervals=True, anomalies=True)

    # Exponential MA
    exp_ma = ExponentialMovingAverage(data)
    exp_ma.basic_smoothing(alpha=0.05)
    exp_ma.plot_basic()

    # Double Exponential MA
    exp_ma.double_smoothing(alpha=0.05, beta=0.5)
    exp_ma.plot_double()


if __name__ == '__main__':
    main()
