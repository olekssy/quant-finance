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


"""Option pricing models Black-Scholes, Monte-Carlo simulation"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

warnings.filterwarnings('ignore')


class EuropeanOption:
    def __init__(self, s0, r, t):
        self.s0 = s0
        self.r = r
        self.t = t
        self.dt = np.exp(-r * t / 365)

    def price_mc(self, runs, mu, sigma, k, call=True):
        mc_p = []
        for _i in range(runs):
            st = asset_price(self.s0, self.t, mu, sigma)[-1][0]
            p = max(0, (-1) ** call * (k - st)) * self.dt
            mc_p.append(p)
        return np.mean(mc_p)

    def price_bs(self, sigma, k):
        r = self.r * self.t / 365
        std = sigma * np.sqrt(self.t / 365)
        kdt = k * np.exp(-r)

        d1 = (np.log(self.s0 / k) + r + 0.5 * std ** 2) / std
        d2 = (np.log(self.s0 / k) + r - 0.5 * std ** 2) / std

        call = self.s0 * st.norm.cdf(d1, 0.0, 1) \
               - kdt * st.norm.cdf(d2, 0.0, 1)
        put = kdt * st.norm.cdf(-d2, 0.0, 1) \
              - self.s0 * st.norm.cdf(-d1, 0.0, 1)
        return call, put

    def price_reportfolio(self):
        pass

    def parity(self, k, call=None, put=None):
        # call + Kdt = put + S0
        if isinstance(call, float):
            put = call + self.dt * k - self.s0
            return put
        elif isinstance(put, float):
            call = put - self.dt * k - self.s0
            return call


def asset_price(s0, t, mu, sigma, plot=False):
    mu = mu / 365
    sigma = sigma / np.sqrt(365)
    origin = np.zeros((1, 1))
    walk = np.random.choice(a=[(mu - sigma), (mu + sigma)], size=(t, 1))
    path = np.concatenate([origin, walk]).cumsum(0) + 1
    price = path * s0

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(t + 1), price, s=0.05)
        ax.plot(price, c='blue', lw=0.25)

        plt.title(f'Underlying asset price evolution'
                  f'\nT = {t}, N({mu:.4f}, {sigma:.4f})')
        plt.tight_layout()
        plt.show()

    return price


def mean_abs_percentage_error(x, y):
    """
    Mean absolute percentage error
    """
    error = np.mean(np.abs((x - y) / y)) * 100
    if np.isnan(error):
        error = 0.0
    return error


def sensitivity_k(option, mu, sigma, runs=1000, step=10, plot=False):
    # Price options for different K
    k = np.arange(900, 1100 + step, step)
    dim = len(k)
    call_mc = np.zeros(dim)
    call_bs = np.zeros(dim)
    put_mc = np.zeros(dim)
    put_bs = np.zeros(dim)
    error = np.zeros(dim)

    print(f'\n==Simulation results of option price[K]==')
    print('{:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s}'.format(
        'K', 'call_mc', 'call_bs', 'put_mc', 'put_bs', 'error, %'))

    for _i in range(dim):
        call_mc[_i] = option.price_mc(runs, mu, sigma, k[_i])
        put_mc[_i] = option.parity(k[_i], call=call_mc[_i])
        call_bs[_i], put_bs[_i] = option.price_bs(sigma, k[_i])
        if call_mc[_i] > put_mc[_i]:
            error[_i] = mean_abs_percentage_error(call_mc[_i], call_bs[_i])
        else:
            error[_i] = mean_abs_percentage_error(put_mc[_i], put_bs[_i])
        print(
            f'{k[_i]:>6n} | {call_mc[_i]:>8.4f} | {call_bs[_i]:>8.4f} | {put_mc[_i]:>8.4f} | {put_bs[_i]:>8.4f} | {error[_i]:>8.4f}')

    mean_error = np.mean(error)
    print(f'Mean absolute percentage error = {mean_error:.4f}%')

    if plot:
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.plot(k, call_mc, label='Call(Monte-Carlo)')
        plt.plot(k, put_mc, label='Put(Monte-Carlo)')
        plt.plot(k, call_bs, label='Call(Black-Scholes)')
        plt.plot(k, put_bs, label='Put(Black-Scholes)')
        plt.xlabel('Strike')
        plt.ylabel('Option price')
        plt.title(f'European option pricing \nruns={runs}, error={mean_error:.4f}%')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.grid(True)
        plt.show()


def sensitivity_sigma(option, mu, k, runs=1000, step=0.001, plot=False):
    # Price options for different K
    sigma = np.arange(0, 0.1 + step, step)
    dim = len(sigma)
    call_mc = np.zeros(dim)
    call_bs = np.zeros(dim)
    put_mc = np.zeros(dim)
    put_bs = np.zeros(dim)
    error = np.zeros(dim)

    print(f'\n==Simulation results of option price[sigma]==')
    print('{:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s}'.format(
        'sigma', 'call_mc', 'call_bs', 'put_mc', 'put_bs', 'error, %'))

    for _i in range(dim):
        call_mc[_i] = option.price_mc(runs, mu, sigma[_i], k)
        put_mc[_i] = option.parity(k, call=call_mc[_i])
        call_bs[_i], put_bs[_i] = option.price_bs(sigma[_i], k)

        if call_mc[_i] > put_mc[_i]:
            error[_i] = mean_abs_percentage_error(call_mc[_i], call_bs[_i])
        else:
            error[_i] = mean_abs_percentage_error(put_mc[_i], put_bs[_i])
        if not _i % 10:
            print(
                f'{sigma[_i]:>6n} | {call_mc[_i]:>8.4f} | {call_bs[_i]:>8.4f} | {put_mc[_i]:>8.4f} | {put_bs[_i]:>8.4f} | {error[_i]:>8.4f}')

    mean_error = np.mean(error)
    print(f'Mean absolute percentage error = {mean_error:.4f}%')

    if plot:
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.plot(sigma, call_mc, label='Call(Monte-Carlo)')
        # plt.plot(sigma, put_mc, label='Put(Monte-Carlo)')
        plt.plot(sigma, call_bs, label='Call(Black-Scholes)')
        plt.xlabel('Sigma')
        plt.ylabel('Option price')
        plt.title(f'European option pricing \nruns={runs}, error={mean_error:.2f}%')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.grid(True)
        plt.show()


def simulation_error(option, mu, sigma, k, step=100, sim_limit=10000, plot=False):
    runs = np.arange(step, sim_limit + step, step)
    dim = len(runs)
    call_mc = np.zeros(dim)
    call_bs = np.zeros(dim)
    put_mc = np.zeros(dim)
    put_bs = np.zeros(dim)
    error = np.zeros(dim)

    print(f'\n==Estimate error due to number of simulations==')
    print('{:^6s} | {:^8s}'.format('runs', 'error, %'))

    for _i in range(dim):
        call_mc[_i] = option.price_mc(runs[_i], mu, sigma, k)
        put_mc[_i] = option.parity(k, call=call_mc[_i])
        call_bs[_i], put_bs[_i] = option.price_bs(sigma, k)
        if call_mc[_i] > put_mc[_i]:
            error[_i] = mean_abs_percentage_error(call_mc[_i], call_bs[_i])
        else:
            error[_i] = mean_abs_percentage_error(put_mc[_i], put_bs[_i])
        print(f'{runs[_i]:>6n} | {error[_i]:>8.4f}')
    mean_error = np.mean(error)

    if plot:
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.plot(runs, error)
        plt.xlabel('Number of Simulations')
        plt.ylabel('Error, %')
        plt.title(
            f'Monte-Carlo Discretization Error, \nMean absolute percentage error={mean_error:.2f}%')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.grid(True)
        plt.show()


def main():
    # Assumptions
    s0 = 1000  # initial asset price
    k = 1020  # strike
    r = 0.02  # discount rate
    sigma = 0.05  # Wiener process assumptions
    t = 30  # days to maturity
    runs = 1000  # MC simulation

    # Build European options
    opt = EuropeanOption(s0, r, t)

    # Price MC
    # %%time
    call_mc = opt.price_mc(runs, r, sigma, k, call=True)
    put_mc = opt.parity(k, call=call_mc)
    print(f'\n==MC simulation== \ncall = {call_mc:.4f} \nput = {put_mc:.4f}')

    # Price Black-Scholes
    # %%time
    call_bs, put_bs = opt.price_bs(sigma, k)
    print(f'\n==BS model== \ncall = {call_bs:.4f} \nput = {put_bs:.4f}')

    # Estimate discretization error
    error = mean_abs_percentage_error(call_bs, call_mc)
    print(f'\nMean absolute percentage error = {error:.2f}%')

    # Sensitivity analysis of price to strike and volatility
    # %%time
    sensitivity_k(opt, r, sigma, runs=runs, plot=True)
    sensitivity_sigma(opt, r, k, runs=runs, plot=True)

    # Sensitivity of discretization error to number of simulations
    # %%time
    simulation_error(opt, r, sigma, k, step=10, sim_limit=1000, plot=True)


if __name__ == '__main__':
    main()
