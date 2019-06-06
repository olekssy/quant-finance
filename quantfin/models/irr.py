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


"""
Linear asset pricing. YTM optimization problem for ZCB, C-bond, FRN
"""

from itertools import product

import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar


class Bond:
    """
    Default-free linear asset class object with deterministic cash-flow.
    Zero-coupon bond, Coupon bond, Floating-rate note, etc.
    """

    def __init__(self, face_value, maturity, price, coupon_rate=0.0, frn=False):
        self.face_value = face_value
        self.maturity = maturity
        self.price = price
        self.coupon_rate = coupon_rate
        self.frn = frn
        self.cf = None
        self.dt = None

    def cash_flow(self):
        """
        Calculate positive stream of the cash-flow from coupons and face value
        :return: list of CF
        """
        if self.frn:
            self.cf = np.append(self.face_value * self.coupon_rate[:-1],
                                self.face_value * (1 + self.coupon_rate[-1]))
            return self.cf
        else:
            self.cf = np.append(
                np.full(self.maturity - 1, self.face_value * self.coupon_rate),
                self.face_value * (1 + self.coupon_rate))
            return self.cf

    def discount_factors(self, rate):
        """
        Calculate discount factors
        :param rate: discount rate, scalar or list
        :return: if param is scalar, return list of discount factors;
                 matrix of dfs if otherwise
        """
        self.dt = []
        if isinstance(rate, (float, int)):
            for t in range(1, self.maturity + 1):
                self.dt.append(1 / (1 + rate) ** t)
        else:
            for _r in rate:
                dt_i = []
                for t in range(1, self.maturity + 1):
                    dt_i.append(1 / (1 + _r) ** t)
                self.dt.append(dt_i)
            self.dt = np.asmatrix(self.dt)
        return self.dt

    def net_present_value(self):
        """
        Calculates NPV = - price + discounted CF
        :return: scalar NPV if dt is a list; if otherwise;
                 list of NPVs from matrix of dt
        """
        if isinstance(self.dt, list):
            return - self.price \
                   + np.dot(self.cf, self.dt)
        else:
            return - self.price \
                   * np.ones(len(self.dt)) \
                   + np.tensordot(self.dt, self.cf, axes=1)


def solve_brent(bond):
    """
    Find YTM, s.t. NPV => 0
    :param bond: Instance of Bond class
    :return: Target YTM, NPV
    """

    def objective_function(bond, ytm):
        """
        Objective function
        :param bond: Instance of Bond class
        :param ytm: IRR or YTM parameter
        :return: absolute value of NPV
        """
        bond.discount_factors(ytm)
        npv = bond.net_present_value()
        return np.abs(npv)

    # Minimize objective func, subject to ytm
    res = minimize_scalar(lambda ytm: objective_function(bond, ytm),
                          method='Brent')
    bond.discount_factors(res.x)
    return res.x, bond.net_present_value()


def solve_integer(dim, scale, bond):
    """
    Integer optimization problem solver
    :param dim: dimension of the grid
    :param scale: scale of the grid segment (min rate step)
    :param bond: instance of Bond class
    :return: YTM for min NPV
    """
    # Initialize discount rate state vector
    mu = np.geomspace(scale, 2 ** dim * scale, num=dim, endpoint=False)
    mu = np.sort(mu)[::-1]

    # matrix of classical register states of given dimension
    sigma_space = np.asarray(np.asmatrix(list(product(range(2), repeat=dim))))
    sigma = sigma_space[sigma_space[:, 0].argsort()]

    # vector of feasible discount rates
    tau = np.dot(sigma, mu)

    # vector of discount factors
    bond.discount_factors(tau)
    # objective function
    omega = bond.net_present_value()
    i = np.argmin(np.abs(omega))
    return tau[i], omega[i]


def main():
    # Assumptions
    face_value = 1000  # payoff at time T
    maturity = 10  # T, years
    price = 850  # market price at time t0
    coupon_rate = 0.05  # coupon rate of C-bond
    float_rate = np.asarray(randint(1, 10, maturity) / 100)

    # Grid search problem dimensions
    dim, scale = 5, 0.005

    # Build ZCB
    zcb = Bond(face_value, maturity, price)
    zcb.cash_flow()
    # Build C-bond
    c_bond = Bond(face_value, maturity, price, coupon_rate)
    c_bond.cash_flow()
    # Build Floating-rate Note
    frn = Bond(face_value, maturity, price, float_rate, frn=True)
    frn.cash_flow()

    # Display assumptions
    print('\n==Assumptions== '
          '\nFace value = {} '
          '\nMaturity = {} '
          '\nMkt price = {} '
          '\nCoupon rate = {} '
          '\nFloating rates = {}'
          '\n'
          .format(face_value,
                  maturity,
                  price,
                  coupon_rate,
                  float_rate))

    # Solve as scalar univariate function using Brent method
    print('==Brent metod== \nBond\t|   YTM   |   NPV')  # header
    # Solve for ZCB
    ytm, npv = solve_brent(zcb)
    print('ZCB \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
    # Solve for C-bond
    ytm, npv = solve_brent(c_bond)
    print('C-bond \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
    # Solve for FRN
    ytm, npv = solve_brent(frn)
    print('FRN \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

    # Solve as integer grid search problem
    print('\n==Intreger grid search== \nBond\t|   YTM   |   NPV')  # header
    # Solve for ZCB
    ytm, npv = solve_integer(dim, scale, zcb)
    print('ZCB \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
    # Solve for C-bond
    ytm, npv = solve_integer(dim, scale, c_bond)
    print('C-bond \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
    # Solve for FRN
    ytm, npv = solve_integer(dim, scale, frn)
    print('FRN \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
    print('Grid size =', 2 ** dim)


if __name__ == '__main__':
    main()
