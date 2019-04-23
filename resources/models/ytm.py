# Linear asset pricing
# YTM optimization problem for ZCB, C-bond, FRN

from itertools import product

import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar


class CouponBond:
    def __init__(self, face_value, maturity, price, coupon_rate=0.0, FRN=False):
        self.face_value = face_value
        self.maturity = maturity
        self.price = price
        self.coupon_rate = coupon_rate
        self.FRN = FRN

    def cash_flow(self):
        if self.FRN:
            self.cf = np.append(self.face_value * self.coupon_rate[:-1],
                                self.face_value * (1 + self.coupon_rate[-1]))
            return self.cf
        else:
            self.cf = np.append(np.full(self.maturity - 1, self.face_value * self.coupon_rate),
                                self.face_value * (1 + self.coupon_rate))
            return self.cf

    def discount_factors(self, rate):
        self.dt = []
        for t in range(1, self.maturity + 1):
            self.dt.append(1 / (1 + rate) ** t)
        return np.array(self.dt)

    def net_present_value(self, price):
        return - price + np.tensordot(self.cf, self.dt, axes=1)


# Construct universal solver
def solve_brent(bond):
    # Construct objective functions
    def objective_function(bond, ytm):
        bond.discount_factors(ytm)
        npv = bond.net_present_value(bond.price)
        return np.abs(npv)
    # Minimize object func, subject to ytm
    res = minimize_scalar(lambda ytm: objective_function(bond, ytm))
    bond.discount_factors(res.x)
    return res.x, bond.net_present_value(bond.price)


# Init range of feasible YTM states
sample = 5
start = 0.005
stop = 0.03
combinations = 2 ** sample


def rate(decision):
    rate_range = np.linspace(start, stop, num=sample)
    a = range(2)
    variable = list(product(a, a, a, a, a))
    return np.tensordot(rate_range, list(variable[decision]), axes=1)


# Construct integer solver
def solve_integer(bond):
    # Construct objective functions
    def objective_function(bond, decision):
        ytm = rate(int(decision))
        bond.discount_factors(ytm)
        npv = bond.net_present_value(bond.price)
        return np.abs(npv)
    # Apply grid search
    best_case = float('inf')
    for decision in range(combinations):
        ytm = rate(decision)
        res = objective_function(bond, decision)
        if best_case > res:
            best_case = res
            best_decision = decision
    ytm = rate(best_decision)
    bond.discount_factors(ytm)
    return ytm, bond.net_present_value(bond.price)
