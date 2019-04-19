import warnings
from itertools import product

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
