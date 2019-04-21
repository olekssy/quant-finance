import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')


class MovingAverage:

    def __init__(self, series):
        self.series = series
        self.window = None
        self.SimpleMovingAverage = None

    def simple(self, window):
        self.window = window
        self.SimpleMovingAverage = self.series.rolling(window=window).mean()
        return self.SimpleMovingAverage

    def plot_simple(self, std_scale=1.96, intervals=False, anomalies=False):
        """
            series - dataframe with timeseries
            window - rolling window size
            plot_intervals - show confidence intervals
            plot_anomalies - show anomalies
        """
        line = self.SimpleMovingAverage

        plt.figure(figsize=(10, 5))
        plt.title("Moving average({})".format(self.window))
        plt.plot(line, "g", label="Rolling mean trend")

        # Plot confidence intervals
        if intervals:
            mae = mean_absolute_error(self.series[self.window:], line[self.window:])
            std = np.std(self.series[self.window:] - line[self.window:])
            lower_bond = line - (mae + std_scale * std)
            upper_bond = line + (mae + std_scale * std)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # plot abnormal values
            if anomalies:
                anomalies = pd.DataFrame(index=self.series.index, columns=self.series.columns)
                anomalies[self.series < lower_bond] = self.series[self.series < lower_bond]
                anomalies[self.series > upper_bond] = self.series[self.series > upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()

    def weighted(self, weights):
        """
            Calculate weighter average on series
        """
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += self.series.iloc[-n - 1] * weights[n]
        return float(result)


class ExponentialMovingAverage(MovingAverage):

    def __init__(self, series):
        super().__init__(series)
        self.alpha = None
        self.beta = None
        self.gamma = None

    def basic_smoothing(self, alpha):
        """
            series - dataset with timestamps
            alpha - float [0.0, 1.0], smoothing parameter
        """
        self.alpha = alpha
        result = [self.series.Open[0]]  # first value is same as series
        for n in range(1, len(self.series)):
            result.append(self.alpha * self.series.Open[n] + (1 - self.alpha) * result[n - 1])
        self.basic_ema = result
        return result

    def plot_basic(self):
        """
            Plots exponential smoothing with different alphas

            series - dataset with timestamps
            alphas - list of floats, smoothing parameters

        """
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 5))

            plt.plot(self.basic_ema, label="Alpha {}".format(self.alpha))

            plt.plot(self.series.values, "c", label="Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing")
            plt.grid(True);
            plt.show()

    def double_smoothing(self, alpha, beta):
        """
            series - dataset with timeseries
            alpha - float [0.0, 1.0], smoothing parameter for level
            beta - float [0.0, 1.0], smoothing parameter for trend
        """
        self.alpha = alpha
        self.beta = beta
        # first value is same as series
        data = self.series.Open
        result = [data[0]]
        for n in range(1, len(data) + 1):
            if n == 1:
                level, trend = data[0], data[1] - data[0]
            if n >= len(data):  # forecasting
                value = result[-1]
            else:
                value = data[n]
            last_level, level = level, self.alpha * value + (1 - self.alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        self.double_ema = result
        return result

    def plot_double(self):
        """
            Plots double exponential smoothing with different alphas and betas

            series - dataset with timestamps
            alphas - list of floats, smoothing parameters for level
            betas - list of floats, smoothing parameters for trend
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
