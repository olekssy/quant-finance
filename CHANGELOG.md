# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2] - 2019-04-16
### Added
- GJR-GARCH model with unsupervised optimization algorithm
- ARIMA unsupervised trend and seasonality detection
- CHANGELOG section for tracking project development
- Directory ./configs/ for storing parameters of models
- Methods to save/load SARIMA configuration from *.csv and *.p files

### Changed
- Seasonal ARIMA model resources, optimization algorithm
- Enhanced graphs of ARIMA
- Appended README and Index with GARCH model application
- Newest on top arrangement of models in Index

### Removed
- ARIMA pickle config files from root directory
- ^TNX.csv from data due to corrupted data

## [0.1.0] - 2019-04-15
### Added
- Seasonal ARIMA model
- Holt-Winters model
- Exponential smoothing methods
- TSA resources to data directory for training models
- MIT license, README section with Index to applications
