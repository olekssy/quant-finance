# Linear asset pricing
# YTM optimization problem for ZCB, C-bond, FRN

from itertools import product

import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar


class CouponBond:
    def __init__(self, face_value, maturity, coupon_rate=0.0, FRN=False):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
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


def main():
    # Solve as scalar univariate function using Brent method
    def brent():
        # Construct universal solver
        def solve_brent(bond):
            # Construct objective functions
            def objective_function(bond, ytm):
                bond.discount_factors(ytm)
                npv = bond.net_present_value(price)
                return np.abs(npv)
            # Minimize object func, subject to ytm
            res = minimize_scalar(lambda ytm: objective_function(bond, ytm))
            bond.discount_factors(res.x)
            return res.x, bond.net_present_value(price)

        print('==Brent metod== \nBond\t|\tYTM\t  |\t  NPV')  # header

        # Solve for ZCB
        ytm, npv = solve_brent(zcb)
        print('ZCB \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

        # Solve for C-bond
        ytm, npv = solve_brent(c_bond)
        print('C-bond \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

        # Solve for FRN
        ytm, npv = solve_brent(frn)
        print('FRN \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

    # Transform optimization problem into binary integer grid search problem
    def integer_grid_search():
        print('\n==Intreger grid search== \nBond\t|\tYTM\t  |\t  NPV')  # header

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
                npv = bond.net_present_value(price)
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
            return ytm, bond.net_present_value(price)

        # Solve for ZCB
        ytm, npv = solve_integer(zcb)
        print('ZCB \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

        # Solve for C-bond
        ytm, npv = solve_integer(c_bond)
        print('C-bond \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))

        # Solve for FRN
        ytm, npv = solve_integer(frn)
        print('FRN \t| {:6.4f}% | {:4.4f}'.format(ytm * 100, npv))
        print('Grid size =', combinations)

    # Assumptions
    face_value = 1000  # payoff at time T
    maturity = 10  # T, years
    price = 850  # market price at time t0
    coupon_rate = 0.05  # coupon rate of C-bond
    float_rate = np.asarray(randint(1, 10, maturity) / 100)  # expected floating coupon rate of FRN

    # Construct ZCB, C-bond and FRN
    # Build ZCB
    zcb = CouponBond(face_value, maturity)
    zcb.cash_flow()

    # Build C-bond
    c_bond = CouponBond(face_value, maturity, coupon_rate)
    c_bond.cash_flow()

    # Build Floating-rate Note
    frn = CouponBond(face_value, maturity, float_rate, FRN=True)
    frn.cash_flow()

    # Display assumptions
    print('\n==Assumptions== \nFace value = {} \nMaturity = {} \nMkt price = {} \nCoupon rate = {} '
          '\nFloating rates = {}\n'.format(face_value, maturity, price, coupon_rate, float_rate))

    # Solve for YTM
    brent()
    integer_grid_search()


if __name__ == '__main__':
    main()
