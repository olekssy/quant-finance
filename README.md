# Quantitative finance models and algorithms

[v0.2.1](CHANGELOG.md)

Collection of models with optimization algorithms for Time series analysis, algorithmic forecasting, quantitative research and risk-management.

## Index
**Optimization models**
1. [IRR](resources/TBD)
    - linear asset pricing: YTM, FX income, capital budgeting, floating-rate notes
    - univariate concave nonlinear optimization via grid search on subintervals
    - mixed integer programming problem, ready-to-use on [NISQ](https://arxiv.org/abs/1801.00862) devices


**Time series analysis models**
1. [GJR-GARCH](resources/GJR-GARCH.ipynb)
    - Glosten-Jagannathan-Runkle GARCH(p, o, q)
    - unsupervised optimization of parameters
    - captures asymmetric shocks (leverage effect)


2. [Seasonal ARIMA](resources/Seasonal-ARIMA.ipynb)
    - ARIMA(p, d, q)x(P, D, Q, s)
    - unsupervised optimization of AR, MA and Seasonal parameters
    - provides one-step-ahead predictions and out-of-sample forecast


3. [Holt-Winters model](resources/Holt-Winters.ipynb)    
    - triple exponential smoothing
    - cross-validation via Conjugate gradient, TNC
    - in-sample prediction and extrapolation


4. [Smoothing methods](resources/Smoothing-Methods.ipynb)
    - Moving average
    - Exponential smoothing
    - Double exponential smoothing

## License and Copyright
Copyright (c) 2019 Oleksii Lialka

Licensed under the [MIT License](LICENSE.md).
