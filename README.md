# MPT Monte Carlo Simulation

### Description

This project implements a Monte Carlo-based portfolio optimization framework using
configurable simulation settings and custom portfolio evaluation functions.
It enables the generation of random portfolios, evaluation of performance metrics,
and extraction of the optimal portfolio based on user-defined criteria.

Currently, the project is limited to the top 10 equity instruments by free float market capitalization of the SPY ETF,
due to the Alpha Vantage API's free tier, which allows only up to 25 requests per day.

The simulation will run under the assumption an environment variable 'ALPHA_VANTAGE_TOKEN' is made available at runtime,
and contains a valid token.
Free tokens can be generated via the [Alpha Vantages site](https://www.alphavantage.co/support/#api-key).


## Features

- **Monte Carlo Simulation**: Run simulations to optimize portfolio weights based on defined metrics (e.g., returns, volatility, Sharpe ratio).
- **Custom Simulation Functions**: Use custom functions to evaluate portfolios with flexibility.
- **Random Portfolio Generation**: Generate random portfolios with user-configurable parameters.
- **Optimal Portfolio Extraction**: Retrieve the optimal portfolio based on a user-selected metric.


### Built With

![Python311](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=fff&style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff&style=for-the-badge)
![Parquet](https://img.shields.io/badge/parquet-0E174B?&style=for-the-badge)


### Version
![Version1](https://img.shields.io/badge/Version-1.0.1-informational?style=flat-square)


### Author

* **Christopher Arnold** - [GitHub](https://github.com/cjarno)
