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
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
from tqdm import tqdm_notebook

warnings.simplefilter('ignore')


def optimize(series, p, o, q, split_date, display=True):
    parameters_list = list(product(p, o, q))

    results = []
    best_aic = float("inf")
    iter_sum = len(parameters_list)
    iter_done = 0

    if display:
        print('\nNumber of combinations = {}\n'.format(iter_sum))
        display_header = True
    for param in tqdm_notebook(parameters_list):

        iter_done += 1
        if display_header:
            print('\n{:<12} {:^16} {:^16}'.format('(p, o, q)', 'AIC', 'Iterations'))
            display_header = False

        try:
            am = arch_model(series, vol='Garch', p=param[0], o=param[1], q=param[2])
            model = am.fit(last_obs=split_date, disp='off', show_warning=False)
        except:
            continue
        aic = model.aic
        if aic < best_aic:
            best_aic = aic
            if display:
                print(param, '\t| {:^16.4f} |{:>8}/{:<8}'.format(aic, iter_done, iter_sum))

        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    p, o, q = result_table.parameters[0]
    print('\nOptimized model GARCH({}, {}, {})\n'.format(p, o, q))
    return p, o, q

def main():
    # Import data
    path = '../data/^GSPC.csv'
    series = pd.read_csv(path, index_col=['Date'], skiprows=None, parse_dates=['Date'])

    # Transform series into returns
    data = series['Adj Close'].pct_change().dropna() * 100

    # Set the limit of learning sample
    n_predict = 100
    split_date = data.index[-n_predict]

    # Initialize GARCH parameters
    p = range(5)
    o = [0, 1]
    q = range(5)

    # Optimize model
    p, o, q = optimize(data, p, o, q, split_date=split_date, display=True)

    # Build model
    model = arch_model(data, vol='Garch', p=p, o=o, q=q)
    results = model.fit(last_obs=split_date, disp='off')
    print(results.summary())

    # Forecast
    print('\nForecasting horizon {} - {}\n'.format(split_date, data.index[-1]))
    forecasts = results.forecast(horizon=3, start=split_date, method='simulation')
    forecasts.variance[split_date:].plot()
    plt.show()


if __name__ == '__main__':
    main()
