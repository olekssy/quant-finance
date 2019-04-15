import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from tqdm import tqdm_notebook

warnings.filterwarnings('ignore')


class ARIMA:
    def __init__(self, series, p, d, q, ps, ds, qs, s):
        self.series = series
        self.p = p
        self.d = d
        self.q = q
        self.ps = ps
        self.ds = ds
        self.qs = qs
        self.s = s

    def parameters_init(self):
        parameters = product(self.p, self.q, self.ps, self.qs)
        parameters_list = list(parameters)
        param_len = len(parameters_list)
        print('Combinations of parameters =', param_len)
        self.parameters_list = parameters_list

    def optimize(self):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order
            s - length of season
        """
        series = self.series
        parameters_list = self.parameters_list
        d = self.d
        D = self.ds
        s = self.s

        results = []
        best_aic = float("inf")

        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model = sm.tsa.statespace.SARIMAX(series, order=(param[0], d, param[1]),
                                                  seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        return result_table

    def plot_predictions(self, model, n_steps):
        """
            Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future

        """
        series = self.series
        s = self.s
        d = self.d

        # adding model values
        data = series.copy()
        data.columns = ['actual']
        data['arima_model'] = model.fittedvalues
        # making a shift on s+d steps, because these values were unobserved by the model
        # due to the differentiating
        data['arima_model'][:s + d] = np.NaN

        # forecasting on n_steps forward
        forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
        forecast = data.arima_model  # .append(forecast)
        # calculate error, again having shifted on s+d steps from the beginning
        error = mean_absolute_percentage_error(data['actual'][s + d:], data['arima_model'][s + d:])

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color='r', label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
        plt.plot(data.actual, label="actual")
        plt.legend()

        plt.grid(True);
        plt.show()


def tsplot(y, lags=None, figsize=(10, 10), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
