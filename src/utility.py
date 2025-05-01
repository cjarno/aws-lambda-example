from loguru import logger


def log_optimal_portfolio(optimal_vector):
    """Log optimal portfolio settings with styled formatting."""
    portfolio_weights_formatted = ", ".join(
        [f"{k}[{v:.4f}]" for k, v in optimal_vector["case"].items()]
    )

    logger.info(
        f"""[Optimal Portfolio Analysis]
        ------------------------------------------------------------
        Portfolio Overview:
            Portfolio Size: {len(optimal_vector['case'])}
        
        Asset Allocation:
        {portfolio_weights_formatted}
            
        ------------------------------------------------------------
        Performance Metrics:
            Returns: {optimal_vector['returns']:.4f}
            Volatility: {optimal_vector['volatility']:.4f}
            Sharpe Ratio: {optimal_vector['sharpe']:.4f}
        ------------------------------------------------------------
        """
    )
