import numpy as np
import csv
from typing import Tuple

def random_normal(size: tuple[int, int] | int, mean: float = 0.0, std: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Generate a random number from a normal distribution.

    Args:
        size (tuple[int, int] | int): The shape of the output array.
        mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
        std (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a normal distribution with mean `mean` and standard deviation `std`.
    """
    
    rng = np.random.default_rng(seed)
    return rng.normal(size = size, loc=mean, scale=std)


def random_uniform(size: int | tuple[int, int], range: tuple[float, float] = (0.0, 1.0), seed: int | None = None) -> np.ndarray:
    """Generate a random number from a uniform distribution.

    Args:
        size (int | tuple[int, int]): The size or shape of the output array.
        range (tuple(float, float), optional): The range of the uniform distribution. Defaults to (0.0, 1.0).
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a uniform distribution over the specified range.
    """
    
    rng = np.random.default_rng(seed)
    return rng.uniform(size=size, low=range[0], high=range[1])

def random_bernoulli(size: int | tuple[int, int], p: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Generate a random number from a Bernoulli distribution.

    Args:
        size (int | tuple[int, int]): The size or shape of the output array.
        p (float, optional): The probability of success. Defaults to 0.5.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a Bernoulli distribution with probability `p`.
    """
    
    rng = np.random.default_rng(seed)
    return rng.binomial(size=size, n=1, p=p)

def paths_to_returns(paths: np.ndarray) -> np.ndarray:
    """Convert price paths to returns.

    Args:
        paths (np.ndarray): A 2D array of shape (paths, N+1) containing the simulated paths of the GBM process.

    Returns:
        np.ndarray: A 1D array of shape (paths,) containing the aggregate returns for each path.
    """
    return (paths[:, -1] - paths[:, 0]) / paths[:, 0]


def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate the Value at Risk (VaR) of a portfolio.

    Args:
        returns (np.ndarray): An array of portfolio returns.
        confidence_level (float, optional): The confidence level for VaR calculation. Defaults to 0.95.

    Returns:
        float: The Value at Risk (VaR) at the specified confidence level.
    """
    
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    
    return -np.percentile(returns, (1 - confidence_level) * 100)



def csv_to_numpy_with_dates(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSV file with columns Date,Open,High,Low,Close,Adj Close,Volume into a numpy array.
    Returns a tuple (dates, data) where dates is a numpy array of date strings and data is a 2D numpy array of floats.
    Each row corresponds to a timestep (date).
    """
    dates = []
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row['Date'])
            # Extract Open, High, Low, Close, Adj Close, Volume as floats
            data.append([
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                float(row['Adj Close']),
                float(row['Volume'])
            ])
    return np.array(dates), np.array(data, dtype=float)


def extract_daily_open(data: np.ndarray) -> np.ndarray:
    """Given a 2D numpy array as returned by csv_to_numpy_with_dates (data),
    return a 1D numpy array of the daily open prices.
    """
    return data[:, 0]


def calibrate_from_csv(
    filepath: str,
    trading_days: int = 252,
    use_years: float | None = None,
) -> tuple[float, float, float]:
    """Calibrate annualised GBM parameters (mu, sigma) from a historical CSV.

    Reads one of Nihan's equity CSVs, computes log-returns on daily open
    prices, and returns annualised drift and volatility that can be passed
    directly into simulate_gbm() or used to build an assets list for
    simulate_portfolio().

    Args:
        filepath (str): Path to a CSV file with columns
            Date, Open, High, Low, Close, Adj Close, Volume.
        trading_days (int): Number of trading days per year used for
            annualisation. Defaults to 252.
        use_years (float | None): If given, only the most-recent
            ``use_years`` worth of data is used for calibration.
            Defaults to None (use all available data).

    Returns:
        tuple[float, float, float]: ``(mu, sigma, S0)`` where
            - ``mu``    – annualised drift (arithmetic mean of log-returns
                          scaled to one year, Itô-corrected to give the
                          GBM drift parameter)
            - ``sigma`` – annualised volatility (std of log-returns scaled
                          to one year)
            - ``S0``    – most-recent open price (convenient starting price
                          for a forward simulation)

    Raises:
        ValueError: If the CSV contains fewer than 2 data points after
            applying the ``use_years`` filter.

    Example::

        mu, sigma, S0 = calibrate_from_csv("data/AAPL_daily_5y.csv")
        paths = simulate_gbm(S0=S0, mu=mu, sigma=sigma)
    """
    _, data = csv_to_numpy_with_dates(filepath)
    open_prices = extract_daily_open(data)  # 1-D array, chronological order

    if use_years is not None:
        n_days = int(use_years * trading_days)
        open_prices = open_prices[-n_days:]

    if len(open_prices) < 2:
        raise ValueError(
            f"Not enough data in {filepath} to calibrate parameters "
            f"(need at least 2 prices, got {len(open_prices)})."
        )

    # Log-returns (daily)
    log_returns = np.diff(np.log(open_prices))

    # Annualise: mean and std of daily log-returns × √T scaling
    daily_mean = float(np.mean(log_returns))
    daily_std = float(np.std(log_returns, ddof=1))

    # GBM drift parameter µ incorporates the Itô correction:
    #   µ = annualised_mean + 0.5 * sigma²
    sigma = daily_std * np.sqrt(trading_days)
    mu = daily_mean * trading_days + 0.5 * sigma ** 2

    S0 = float(open_prices[-1])

    return mu, sigma, S0


def calibrate_portfolio_from_csvs(
    csv_paths: list[str],
    weights: list[float] | None = None,
    trading_days: int = 252,
    use_years: float | None = None,
) -> list[tuple[float, float, float]]:
    """Calibrate GBM parameters for multiple equities from their CSV files.

    Convenience wrapper around calibrate_from_csv that builds the
    ``assets`` list expected by simulate_portfolio() directly from
    Nihan's CSV files.

    Args:
        csv_paths (list[str]): Ordered list of CSV file paths, one per equity.
        weights (list[float] | None): Portfolio weights in the same order as
            ``csv_paths``.  If None, equal weights are used.
        trading_days (int): Trading days per year for annualisation.
            Defaults to 252.
        use_years (float | None): Limit calibration to the most-recent N
            years of data per file.  Defaults to None (all data).

    Returns:
        list[tuple[float, float, float]]: List of ``(mu, sigma, weight)``
            tuples ready to pass into simulate_portfolio().

    Raises:
        ValueError: If ``weights`` is provided but its length differs from
            ``csv_paths``.

    Example::

        import glob
        csvs = sorted(glob.glob("data/*_daily_5y.csv"))
        assets = calibrate_portfolio_from_csvs(csvs)
        port_returns = simulate_portfolio(assets)
    """
    if weights is not None and len(weights) != len(csv_paths):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match "
            f"length of csv_paths ({len(csv_paths)})."
        )

    if weights is None:
        weights = [1.0 / len(csv_paths)] * len(csv_paths)

    assets = []
    for path, w in zip(csv_paths, weights):
        mu, sigma, _S0 = calibrate_from_csv(
            path, trading_days=trading_days, use_years=use_years
        )
        assets.append((mu, sigma, w))

    return assets

