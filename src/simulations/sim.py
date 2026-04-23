from models.gbm import GBM
from utils.util import value_at_risk, paths_to_returns, extract_daily_open, csv_to_numpy_with_dates
import matplotlib.pyplot as plt
import numpy as np


def compare_hist_to_gbm(file: str, show: bool = True) -> None:
    """Compare historical stock price data to a simulated GBM path.

    Args:
        file (str): The path to the CSV file containing historical stock price data.
        show (bool, optional): Whether to display the plot. Defaults to True.
    """
    
    # Load historical data
    dates, data = csv_to_numpy_with_dates(file)
    open_prices = extract_daily_open(data)
    print(f"Loaded {len(dates)} historical data points from {file}.")


    #for debug
    print(f"First 5 dates: {dates[:5]}")
    print(f"First 5 open prices: {open_prices[:5]}")

    # Use the last 4 years for mean/volatility, predict the last year
    trading_days = 252
    total_days = len(open_prices)
    if total_days < 5 * trading_days:
        raise ValueError("Not enough data for 5 years (need at least 1260 trading days)")

    # Indices for last 5 years
    idx_4y_start = total_days - trading_days * 5
    idx_1y_start = total_days - trading_days

    # Use years 1-4 for stats, year 5 for prediction
    prices_4y = open_prices[idx_4y_start:idx_1y_start]
    returns_4y = np.diff(prices_4y) / prices_4y[:-1]
    mean_return = float(np.mean(returns_4y) * trading_days)
    volatility = float(np.std(returns_4y) * np.sqrt(trading_days))
    print(f"Calculated mean return (annualized): {mean_return:.4f}, volatility (annualized): {volatility:.4f} from last 4 years.")

    # Simulate GBM for the last year
    S0 = open_prices[idx_1y_start]
    gbm = GBM(S0=S0, mu=mean_return, sigma=volatility, T=1.0, N=trading_days)
    simulated_paths = gbm.simulate(paths=500, show=False)

    

    # Plot historical vs simulated
    if show:
        plt.figure(figsize=(12, 6))
        # Use a colormap for colorful faded paths
        cmap = plt.get_cmap('tab20', simulated_paths.shape[0])
        for i in range(simulated_paths.shape[0]):
            plt.plot(
                dates[idx_1y_start:],
                simulated_paths[i][1:],
                color=cmap(i),
                alpha=0.15,
                linewidth=1
            )
        # Plot the expected value (mean path)
        mean_path = np.mean(simulated_paths, axis=0)
        plt.plot(dates[idx_1y_start:], mean_path[1:], color='tab:red', alpha=0.7, linewidth=2, label='Expected Value (Mean Path)')
        # Plot actual last year prices
        plt.plot(dates[idx_1y_start:], open_prices[idx_1y_start:], label='Historical Last Year', color='tab:blue')
        # Plot all simulated paths
        plt.title('Historical Last Year vs Simulated GBM Paths')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        # Make x-axis ticks sparse and format dates
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[::21])  # Show roughly every month (21 trading days)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        plt.tight_layout()
        plt.show()

