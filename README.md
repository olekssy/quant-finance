# Time Series Analysis models

[v0.2](CHANGELOG.md)

Collection of models with optimization algorithms for Time series analysis, algorithmic forecasting and quantitative research.

Models are trained and tested on S&P 500 historical prices.

## Index

**Currently available models and algorithms:**

1. [GJR-GARCH](GJR-GARCH.ipynb)
    - Glosten-Jagannathan-Runkle GARCH(p, o, q)
    - unsupervised optimization of parameters
    - captures asymmetric shocks (leverage effect) 
2. [Seasonal ARIMA](Seasonal-ARIMA.ipynb)
    - ARIMA(p, d, q)x(P, D, Q, s)
    - unsupervised optimization of AR, MA and Seasonal parameters
    - provides one-step-ahead predictions and out-of-sample forecast
3. [Holt-Winters model](Holt-Winters.ipynb)    
    - triple exponential smoothing
    - cross-validation via Conjugate gradient, TNC
    - in-sample prediction and extrapolation
4. [Smoothing methods](Smoothing-Methods.ipynb)
    - Moving average
    - Exponential smoothing
    - Double exponential smoothing

## License and Copyright
Copyright (c) 2019 Oleksii Lialka

Licensed under the [MIT License](LICENSE.md).