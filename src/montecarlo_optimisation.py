import os
import requests
import datetime as dt
from loguru import logger
from pathlib import Path
from enum import Enum
from typing import Union, Optional, List, Tuple, Dict
from dotenv import load_dotenv

import pandas as pd
import numpy as np

# todo: will move to main
load_dotenv()


class AlphaVantageFunctionEndpoint(Enum):
    etf_profile = 'ETF_PROFILE'
    daily_timeseries = 'TIME_SERIES_DAILY'
    weekly_adj_timeseries = 'TIME_SERIES_WEEKLY_ADJUSTED'
    treasury_yield = 'TREASURY_YIELD'


class AlphaVantageOutputSize(Enum):
    compact = 'compact'
    full = 'full'


class AlphaVantageInterval(Enum):
    daily = 'daily'


class AlphaVantageTreasuryMaturity(Enum):
    ten_year = '10year'


# https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey=demo


def generate_alphavantage_function_url(
        token: str,
        endpoint: AlphaVantageFunctionEndpoint,
        base_url: str = 'https://www.alphavantage.co',
        symbol: Optional[str] = None,
        interval: Optional[AlphaVantageInterval] = None,
        maturity: Optional[AlphaVantageTreasuryMaturity] = None,
        outputsize: Optional[AlphaVantageOutputSize] = None
) -> str:
    """
    Generates the full url for an AlphaVantage endpoint.

    Follows the generic form of AlphaVantage API structures,
    where a target 'function' and 'symbol' are required.

    More complex forms are required for other endpoints.

    """
    url_ex_key = fr'{base_url}/query?function={endpoint.value}'
    if symbol:
        url_ex_key += fr'&symbol={symbol}'
    if interval:
        url_ex_key += fr'&interval={interval.value}'
    if maturity:
        url_ex_key += fr'&maturity={maturity.value}'
    if outputsize:
        url_ex_key += fr'&outputsize={outputsize.value}'
    logger.trace(url_ex_key)
    return fr'{url_ex_key}&apikey={token}'


portfolio_url = generate_alphavantage_function_url(
    token=os.getenv('ALPHA_VANTAGE_TOKEN'),
    endpoint=AlphaVantageFunctionEndpoint.etf_profile,
    symbol='SPY',
)

rfr_url = generate_alphavantage_function_url(
    token=os.getenv('ALPHA_VANTAGE_TOKEN'),
    endpoint=AlphaVantageFunctionEndpoint.treasury_yield,
    maturity=AlphaVantageTreasuryMaturity.ten_year,
    interval=AlphaVantageInterval.daily
)


def load_latest_weekly_timeseries(sym_url_dict: Dict[str, str]) -> pd.DataFrame:
    """
    """
    cwd = Path.cwd()
    cache_filepath = cwd.joinpath(f'_{dt.datetime.now().strftime("%Y%m%d")}_timeseries_cache.parquet')

    if not cache_filepath.exists():
        logger.info(f'cache unavailable, downloading and processing data to cache {cache_filepath}')

        all_ts = []
        for _sym, _url in sym_url_dict.items():
            # todo: add success/failure based on returned status code
            response = requests.get(url=_url)

            if not response.status_code == 200:
                logger.warning(f"Unexpected Response: [{response.status_code}] {response.reason}")

            response_json = response.json()
            ts_df = pd.DataFrame(response_json['Weekly Adjusted Time Series']).T
            ts_close_series = ts_df['5. adjusted close']
            ts_close_series.name = _sym
            all_ts.append(ts_close_series)

        final_df = pd.concat(all_ts, axis=1)
        final_df.reset_index(inplace=True)
        final_df.rename(columns={'index': 'weekly_close_date'}, inplace=True)
        final_df.to_parquet(cache_filepath, index=False)
    else:
        logger.info(f'cache available, loading from cache {cache_filepath}')
        final_df = pd.read_parquet(cache_filepath)
    final_df.set_index(keys='weekly_close_date', inplace=True)
    final_df.index = pd.to_datetime(final_df.index)
    return final_df


def load_latest_portfolio(portfolio_url: str) -> pd.DataFrame:
    """
    """
    cwd = Path.cwd()
    cache_filepath = cwd.joinpath(f'_{dt.datetime.now().strftime("%Y%m%d")}_holdings_cache.parquet')

    if not cache_filepath.exists():
        logger.info(f'cache unavailable, downloading and processing data to cache {cache_filepath}')
        # todo: add success/failure based on returned status code
        response = requests.get(url=portfolio_url)
        if not response.status_code == 200:
            logger.warning(f"Unexpected Response: [{response.status_code}] {response.reason}")

        df = pd.DataFrame(response.json()['holdings']).sort_values(by='weight', ascending=False)
        df.to_parquet(cache_filepath, index=False)
    else:
        logger.info(f'cache available, loading from cache {cache_filepath}')
        df = pd.read_parquet(cache_filepath)
    return df


def load_latest_rfr_timeseries(url) -> pd.DataFrame:
    cwd = Path.cwd()
    cache_filepath = cwd.joinpath(f'_{dt.datetime.now().strftime("%Y%m%d")}_rfr_cache.parquet')
    if not cache_filepath.exists():
        logger.info(f'cache unavailable, downloading and processing data to cache {cache_filepath}')
        # todo: add success/failure based on returned status code
        response = requests.get(url=url)
        if not response.status_code == 200:
            logger.warning(f"Unexpected Response: [{response.status_code}] {response.reason}")
        df = pd.DataFrame(response.json()['data'])
        df.to_parquet(cache_filepath, index=False)
    else:
        logger.info(f'cache available, loading from cache {cache_filepath}')
        df = pd.read_parquet(cache_filepath)
    df.set_index(keys='date', inplace=True)
    df.rename(columns={'value': 'rfr'}, inplace=True)
    df.index = pd.to_datetime(df.index)
    df['rfr'] = pd.to_numeric(df['rfr'], errors='coerce')
    df['rfr'] = df['rfr'] / 100
    df['rfr'] = df['rfr'].interpolate(method='quadratic')
    return df


df_portfolio = load_latest_portfolio(
    portfolio_url=portfolio_url,
)

df_rfr = load_latest_rfr_timeseries(
    url=rfr_url
)

df_portfolio_top10 = df_portfolio.iloc[0:10].copy()

urls_dict = {
    sym: generate_alphavantage_function_url(
        token=os.getenv('ALPHA_VANTAGE_TOKEN'),
        symbol=sym,
        endpoint=AlphaVantageFunctionEndpoint.weekly_adj_timeseries,
        # outputsize=AlphaVantageOutputSize.compact,
    )
    for sym
    in df_portfolio_top10['symbol'].values
}

df_timeseries = load_latest_weekly_timeseries(
    sym_url_dict=urls_dict,
)

last_valid_tick = df_timeseries.apply(pd.Series.last_valid_index).max()
df_timeseries_valid_ticks = df_timeseries[df_timeseries.index >= last_valid_tick]
df_timeseries_valid_ticks = df_timeseries_valid_ticks.astype(float)
df_log_returns_timeseries = np.log(df_timeseries_valid_ticks / df_timeseries_valid_ticks.shift(-1)).iloc[:-1]
# df_returns_timeseries = df_timeseries_valid_ticks.pct_change(-1).iloc[:-1]

df_log_returns_rfr_timeseries = pd.merge(left=df_log_returns_timeseries, right=df_rfr, left_index=True,
                                         right_index=True, how='left')
df_log_returns_rfr_timeseries.bfill(inplace=True)

na_value_count = df_log_returns_rfr_timeseries.isna().sum().sum()
if na_value_count > 0:
    logger.info(
        f"Data was not able to be completely corrected for missing values. Currently {na_value_count} NAs exist.")
else:
    logger.success('Data retrieved and prepared..')

# Annualised mean returns
annualised_mean_returns = df_log_returns_timeseries.mean() * 52

# Annualised covariance matrix
annualised_covariance_matrix = df_log_returns_timeseries.cov() * 52

# Annualised mean rfr
annualised_mean_rfr = df_rfr.mean().iloc[0]


class MonteCarloMethod:
    def __init__(self, mean_rfr, mean_returns: pd.Series, covariance_matrix: pd.DataFrame, iterations: int = 1000):
        self.iterations = iterations
        self.n_assets = len(mean_returns)
        self.case_weights = self._generate_random_weights()
        self.results = np.zeros((3, iterations))

        self.mean_rfr = mean_rfr
        self.mean_returns = mean_returns
        self.covariance_matrix = covariance_matrix

    def _generate_random_weights(self):
        rng = np.random.default_rng()
        case_weights = rng.random((self.iterations, self.n_assets))
        normalised_case_weights = case_weights / case_weights.sum(axis=1).reshape(-1, 1)
        return normalised_case_weights

    def run(self):
        for idx in range(self.iterations):
            case_weights = self.case_weights[idx, :]
            portfolio_return = np.dot(case_weights, self.mean_returns)
            portfolio_volatility = np.sqrt(case_weights.T @ self.covariance_matrix @ case_weights)
            portfolio_sharpe = self._calc_sharpe_ratio(portfolio_return, portfolio_volatility)
            self.results[:, idx] = [portfolio_return, portfolio_volatility, portfolio_sharpe]
        return self.results

    def _calc_sharpe_ratio(self, portfolio_returns, portfolio_volatility):
        portfolio_sharpe = (portfolio_returns - self.mean_rfr) / portfolio_volatility
        return portfolio_sharpe

    def get_optimal_portfolio(self):
        idx = np.argmax(self.results[2, :])

        return {
            'weights': self.case_weights[idx, :],
            'returns': self.results[0, idx],
            'volatility': self.results[1, idx],
            'sharpe': self.results[2, idx],
        }


mcm = MonteCarloMethod(
    mean_rfr=annualised_mean_rfr,
    mean_returns=annualised_mean_returns,
    covariance_matrix=annualised_covariance_matrix,
    iterations=1000)

iteration_result = mcm.run()

opt_port = mcm.get_optimal_portfolio()

logger.info('done')
