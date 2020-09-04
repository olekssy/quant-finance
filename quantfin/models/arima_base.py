"""
Stock price forecasting with ARIMA for 45 min coding challenge interview


## Assumptions
1/ Manual h-parameters tuning (p, d, q)
2/ Opimized rolling window retraining algorithm
~O(N^2) time complexity (~15 min on Xeon E5 3.2GHz)
3/ Diferencing approach for de-trending and stationary transformation
4/ No Brutlag's predicted deviation estimates


## Result
=== Performance metrics ===
R^2 (1.0 is the best) = 0.9987
MSE (0.0 is the best) = 0.2351
MAE (0.0 is the best) = 0.2003
N = 2265
"""


import warnings

import pandas as pd
import sklearn.metrics
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -------------------- Assumptions --------------------
ticker = 'AAPL'
pdq = (2, 1, 2)  # ARIMA(p, d, q) params
p_value = 0.05  # critical value threshold
train_size = 0.10  # size of initial train sample for one-step-ahead forecast
verbose = True


# -------------------- Helper Functions --------------------
def download_finseries(ticker):
    """ Collect historical stock price from YF """
    import pandas_datareader

    # retrieve data from YF
    finseries = pandas_datareader.data.DataReader(
        name=ticker,
        data_source='yahoo',
        start='2010-01-01',
        end='2020-01-01')

    # save to csv
    finseries['Adj Close'].to_csv(''.join([ticker, '.csv']))
    return finseries


def adf_test(series, verbose=False):
    """ Augmented Dickey-Fuller Test for testing difference-stationarity """
    adf_report = adfuller(series)
    adf_p_value = adf_report[1]

    if verbose:
        if adf_p_value > p_value:
            print('Series are NOT difference-stationary. Apply differencing.')
        else:
            print('Series are difference stationary.')
        print(f'p-value: {adf_p_value:4.2f}')
    return adf_p_value


def kpss_test(series, verbose=False):
    """ KPSS test for for testing trend-stationarity """
    kpss_report = kpss(series, nlags="auto")
    kpss_p_value = kpss_report[1]

    if verbose:
        if kpss_p_value > p_value:
            print('Series are NOT trend-stationary. Remove trend.')
        else:
            print('Series are trend-stationary.')
        print(f'p-value: {kpss_p_value:4.2f}')
    return kpss_p_value


def difference(series, d=1):
    """ Shift difference series to remove trend """
    return series.diff(d).dropna()


def evaluate_model(true, pred, verbose):
    """ Evalute predictive power of the model """
    MSE = sklearn.metrics.mean_squared_error(true, pred)
    R2 = sklearn.metrics.r2_score(true, pred)
    MAE = sklearn.metrics.median_absolute_error(true, pred)

    if verbose:
        print('=== Performance metrics ===')
        print(f'R^2 (1.0 is the best) = {R2:4.4f}')
        print(f'MSE (0.0 is the best) = {MSE:4.4f}')
        print(f'MAE (0.0 is the best) = {MAE:4.4f}\n')


# -------------------- Model Settings --------------------
def train_model(series, pdq, verbose=False):
    """ Train ARIMA model """
    model = ARIMA(series.values, order=pdq)
    model = model.fit()
    if verbose:
        print(model.summary())
    return model


def rolling_forecast(series, pdq, train_size=0.2, verbose=False):
    """ Make one-step-ahead forecast w/ ARIMA """
    # find right index of train size
    train_id = int(series.shape[0]*train_size)

    # make one step ahead forecast starting from the end of the train sample
    forecast = []
    for i in tqdm(range(train_id, series.shape[0])):
        # train model
        model = train_model(series[:i], pdq, verbose=False)
        # predict
        forecast.append(model.forecast()[0])
    forecast = pd.Series(forecast, index=series.index[train_id:])

    # write forecast-actual to df
    result_df = pd.DataFrame(
        {
            'true': series['Adj Close'].values[train_id:].tolist(),
            'pred': forecast.values
        },
        index=series.index[train_id:]
    )
    return result_df


def main():
    # collect data
    finseries = download_finseries(ticker)

    # test stationarity
    adf_test(finseries, verbose)
    kpss_test(finseries, verbose)

    # find d-parameter
    # finseries = difference(finseries, d=1)
    # repeat stationarity test
    # adf_p_value = adf_test(finseries, verbose)
    # kpss_p_value = kpss_test(train, verbose)

    # make one-step-ahead forecast
    result = rolling_forecast(
        series=finseries,
        pdq=pdq,
        train_size=train_size,
        verbose=False
    )

    # performance metrics
    evaluate_model(result['true'], result['pred'], verbose)


if __name__ == '__main__':
    main()
