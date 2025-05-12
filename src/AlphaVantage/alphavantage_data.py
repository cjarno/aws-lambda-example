import os
import requests
import datetime as dt
from loguru import logger
from pathlib import Path
from enum import Enum
from typing import Dict

import pandas as pd
import numpy as np


def load_top10_annualised_data(cache_subdir: str = ".cache"):
    """Loads and processes annualized financial data for portfolio optimization.

    Retrieves portfolio holdings, risk-free rate (RFR), and weekly adjusted time series
    data from Alpha Vantage APIs, processes them into log returns, and computes annualized
    mean returns, covariance matrix, and mean RFR.

    Data is cached locally to avoid redundant API calls within the same day.

    Limited to the top 10 holdings by market capitalisation due to the Alpha Vantage APIs free tier
    allowing up to 25 requests per day.

    Args:
        cache_subdir (str): The target subdirectory of the project to store the cached parquet files.

    Returns:
        dict: A dictionary containing:
            * mean_returns (pd.Series): Annualized mean log returns for top 10 holdings,
              shape (10,).
            * covariance_matrix (pd.DataFrame): Annualized covariance matrix of log returns,
              shape (10, 10).
            * mean_rfr (float): Annualized mean risk-free rate.

    Raises:
        requests.exceptions.RequestException: If an API request fails.
        KeyError: If the API response lacks expected keys or data.
        ValueError: If data processing results in invalid or missing values.
    """

    cache_dir_path = Path.cwd().joinpath(cache_subdir)
    cache_dir_path.mkdir(exist_ok=True)

    class AlphaVantageFunctionEndpoint(Enum):
        """Enum for Alpha Vantage API function endpoints."""

        etf_profile = "ETF_PROFILE"
        daily_timeseries = "TIME_SERIES_DAILY"
        weekly_adj_timeseries = "TIME_SERIES_WEEKLY_ADJUSTED"
        treasury_yield = "TREASURY_YIELD"

    class AlphaVantageOutputSize(Enum):
        """Enum for Alpha Vantage API output size options."""

        compact = "compact"
        full = "full"

    class AlphaVantageInterval(Enum):
        """Enum for Alpha Vantage API time intervals."""

        daily = "daily"

    class AlphaVantageTreasuryMaturity(Enum):
        """Enum for Alpha Vantage API treasury yield maturities."""

        ten_year = "10year"

    def generate_alphavantage_function_url(
        token: str,
        endpoint: AlphaVantageFunctionEndpoint,
        base_url: str = "https://www.alphavantage.co",
        symbol: str | None = None,
        interval: AlphaVantageInterval | None = None,
        maturity: AlphaVantageTreasuryMaturity | None = None,
        output_size: AlphaVantageOutputSize | None = None,
    ) -> str:
        """Generates a URL for an Alpha Vantage API endpoint.

        Constructs a URL based on the specified endpoint and optional parameters,
        following the Alpha Vantage API structure. The API token is appended to the URL.

        Args:
            token (str): API token for Alpha Vantage authentication.
            endpoint (AlphaVantageFunctionEndpoint): The API function endpoint (e.g., ETF_PROFILE, TIME_SERIES_DAILY).
            base_url (str): Base URL for the Alpha Vantage API (default: 'https://www.alphavantage.co').
            symbol (str | None): Ticker symbol for the security (e.g., 'SPY'), if applicable.
            interval (AlphaVantageInterval | None): Time interval for the data (e.g., daily), if applicable.
            maturity (AlphaVantageTreasuryMaturity | None): Treasury yield maturity (e.g., 10year), if applicable.
            output_size (AlphaVantageOutputSize | None): Size of the output data (e.g., compact, full), if applicable.

        Returns:
            str: The complete URL for the API request.

        Raises:
            ValueError: If required parameters for the endpoint are missing or invalid.
        """
        url_ex_key = rf"{base_url}/query?function={endpoint.value}"
        if symbol:
            url_ex_key += rf"&symbol={symbol}"
        if interval:
            url_ex_key += rf"&interval={interval.value}"
        if maturity:
            url_ex_key += rf"&maturity={maturity.value}"
        if output_size:
            url_ex_key += rf"&outputsize={output_size.value}"
        logger.trace(url_ex_key)
        return rf"{url_ex_key}&apikey={token}"

    etf_composition_url = generate_alphavantage_function_url(
        token=os.getenv("ALPHA_VANTAGE_TOKEN"),
        endpoint=AlphaVantageFunctionEndpoint.etf_profile,
        symbol="SPY",
    )

    rfr_url = generate_alphavantage_function_url(
        token=os.getenv("ALPHA_VANTAGE_TOKEN"),
        endpoint=AlphaVantageFunctionEndpoint.treasury_yield,
        maturity=AlphaVantageTreasuryMaturity.ten_year,
        interval=AlphaVantageInterval.daily,
    )

    def load_latest_weekly_timeseries(sym_url_dict: Dict[str, str]) -> pd.DataFrame:
        """Loads weekly adjusted time series data for specified symbols.

        Retrieves weekly adjusted closing prices from Alpha Vantage for the given symbols,
        caching the data locally to avoid redundant API calls within the same day. The data
        is returned as a DataFrame with dates as the index and symbols as columns.

        Args:
            sym_url_dict (Dict[str, str]): Dictionary mapping ticker symbols to their corresponding Alpha
              Vantage API URLs.

        Returns:
            pd.DataFrame: DataFrame with weekly adjusted closing prices,
                indexed by date and columns as symbol tickers.

        Raises:
            requests.exceptions.RequestException: If an API request fails.
            KeyError: If the API response lacks expected keys (e.g., 'Weekly Adjusted Time Series').
            ValueError: If the response data cannot be processed into a valid DataFrame.
        """

        cache_filepath = cache_dir_path.joinpath(
            f'_{dt.datetime.now().strftime("%Y%m%d")}_timeseries_cache.parquet'
        )

        if not cache_filepath.exists():
            logger.info(
                f"cache unavailable, downloading and processing data to cache {cache_filepath}"
            )

            all_ts = []
            for _sym, _url in sym_url_dict.items():
                response = requests.get(url=_url)

                if not response.status_code == 200:
                    logger.warning(
                        f"Unexpected Response: [{response.status_code}] {response.reason}"
                    )

                response_json = response.json()
                ts_df = pd.DataFrame(response_json["Weekly Adjusted Time Series"]).T
                ts_close_series = ts_df["5. adjusted close"]
                ts_close_series.name = _sym
                all_ts.append(ts_close_series)

            final_df = pd.concat(all_ts, axis=1)
            final_df.reset_index(inplace=True)
            final_df.rename(columns={"index": "weekly_close_date"}, inplace=True)
            final_df.to_parquet(cache_filepath, index=False)
        else:
            logger.info(f"cache available, loading from cache {cache_filepath}")
            final_df = pd.read_parquet(cache_filepath)
        final_df.set_index(keys="weekly_close_date", inplace=True)
        final_df.index = pd.to_datetime(final_df.index)
        return final_df

    def load_latest_portfolio(portfolio_url: str) -> pd.DataFrame:
        """Loads portfolio holdings data for a specified ETF.

        Retrieves the holdings data for an ETF (e.g., SPY) from Alpha Vantage, caching
        it locally to avoid redundant API calls within the same day. The data is sorted
        by weight in descending order.

        Args:
            portfolio_url (str): URL for the Alpha Vantage ETF_PROFILE endpoint.

        Returns:
            df (pd.DataFrame): DataFrame containing portfolio holdings, with columns including
              'symbol' and 'weight', sorted by weight in descending order.

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            KeyError: If the API response lacks the 'holdings' key.
            ValueError: If the response data cannot be processed into a valid DataFrame.
        """
        cache_filepath = cache_dir_path.joinpath(
            f'_{dt.datetime.now().strftime("%Y%m%d")}_holdings_cache.parquet'
        )

        if not cache_filepath.exists():
            logger.info(
                f"cache unavailable, downloading and processing data to cache {cache_filepath}"
            )
            response = requests.get(url=portfolio_url)
            if not response.status_code == 200:
                logger.warning(
                    f"Unexpected Response: [{response.status_code}] {response.reason}"
                )

            df = pd.DataFrame(response.json()["holdings"]).sort_values(
                by="weight", ascending=False
            )
            df.to_parquet(cache_filepath, index=False)
        else:
            logger.info(f"cache available, loading from cache {cache_filepath}")
            df = pd.read_parquet(cache_filepath)
        return df

    def load_latest_rfr_timeseries(url: str) -> pd.DataFrame:
        """Loads daily risk-free rate (RFR) time series data.

        Retrieves 10-year treasury yield data from Alpha Vantage, caching it locally to
        avoid redundant API calls within the same day. The data is processed to convert
        yields to decimal form, interpolate missing values, and set a datetime index.

        Args:
            url (str): URL for the Alpha Vantage TREASURY_YIELD endpoint.

        Returns:
            df (pd.DataFrame): DataFrame with a single 'rfr' column (risk-free rate in decimal
              form), indexed by date (pd.DatetimeIndex).

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            KeyError: If the API response lacks the 'data' key.
            ValueError: If the response data cannot be processed or contains invalid values.
        """
        cache_filepath = cache_dir_path.joinpath(
            f'_{dt.datetime.now().strftime("%Y%m%d")}_rfr_cache.parquet'
        )
        if not cache_filepath.exists():
            logger.info(
                f"cache unavailable, downloading and processing data to cache {cache_filepath}"
            )
            response = requests.get(url=url)
            if not response.status_code == 200:
                logger.warning(
                    f"Unexpected Response: [{response.status_code}] {response.reason}"
                )
            df = pd.DataFrame(response.json()["data"])
            df.to_parquet(cache_filepath, index=False)
        else:
            logger.info(f"cache available, loading from cache {cache_filepath}")
            df = pd.read_parquet(cache_filepath)
        df.set_index(keys="date", inplace=True)
        df.rename(columns={"value": "rfr"}, inplace=True)
        df.index = pd.to_datetime(df.index)
        df["rfr"] = pd.to_numeric(df["rfr"], errors="coerce")
        df["rfr"] = df["rfr"] / 100
        df["rfr"] = df["rfr"].interpolate(method="quadratic")
        return df

    df_portfolio = load_latest_portfolio(
        portfolio_url=etf_composition_url,
    )

    df_rfr = load_latest_rfr_timeseries(url=rfr_url)

    df_portfolio_top10 = df_portfolio.iloc[0:10].copy()

    urls_dict = {
        sym: generate_alphavantage_function_url(
            token=os.getenv("ALPHA_VANTAGE_TOKEN"),
            symbol=sym,
            endpoint=AlphaVantageFunctionEndpoint.weekly_adj_timeseries,
        )
        for sym in df_portfolio_top10["symbol"].values
    }

    df_timeseries = load_latest_weekly_timeseries(
        sym_url_dict=urls_dict,
    )

    last_valid_tick = df_timeseries.apply(pd.Series.last_valid_index).max()
    df_timeseries_valid_ticks = df_timeseries[df_timeseries.index >= last_valid_tick]
    df_timeseries_valid_ticks = df_timeseries_valid_ticks.astype(float)
    df_log_returns_timeseries = np.log(
        df_timeseries_valid_ticks / df_timeseries_valid_ticks.shift(-1)
    ).iloc[:-1]

    df_log_returns_rfr_timeseries = pd.merge(
        left=df_log_returns_timeseries,
        right=df_rfr,
        left_index=True,
        right_index=True,
        how="left",
    )
    df_log_returns_rfr_timeseries.bfill(inplace=True)

    na_value_count = df_log_returns_rfr_timeseries.isna().sum().sum()
    if na_value_count > 0:
        logger.info(
            f"Data was not able to be completely corrected for missing values. Currently {na_value_count} NAs exist."
        )
    else:
        logger.success("Data retrieved and prepared.")

    annualised_data = dict(
        instruments=df_portfolio_top10["symbol"].values,
        mean_returns=df_log_returns_timeseries.mean() * 52,
        covariance_matrix=df_log_returns_timeseries.cov() * 52,
        mean_rfr=df_rfr.mean().iloc[0],
    )

    return annualised_data
