from models.gbm import GBM
from utils.util import value_at_risk, paths_to_returns, extract_daily_open, csv_to_numpy_with_dates
import matplotlib.pyplot as plt
import numpy as np
import os



def compare_hist_to_gbm(file: str, num_sims: int = 500, show: bool = True) -> None:
    """Compare historical stock price data to a simulated GBM path.

    Args:
        file (str): The path to the CSV file containing historical stock price data.
        num_sims (int, optional): Number of simulated paths. Defaults to 500.
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
    simulated_paths = gbm.simulate(paths=num_sims, show=False)

    

    # Plot historical vs simulated
    if show:
        plt.figure(figsize=(12, 6))
        # Use a colormap for colorful faded paths
        cmap = plt.get_cmap('tab20', simulated_paths.shape[0])
        simulated_returns = paths_to_returns(simulated_paths)
        var_95 = value_at_risk(simulated_returns, confidence_level=0.95)
        # Find the path closest to the 95% VaR
        var_idx = np.argmin(np.abs(simulated_returns + var_95))
        var_area_color = 'blue'
        # Draw all simulated paths
        for i in range(simulated_paths.shape[0]):
            plt.plot(
                dates[idx_1y_start:],
                simulated_paths[i][1:],
                color=cmap(i) if i != var_idx else cmap(i),
                alpha=0.15,
                linewidth=1
            )

        # Draw a horizontal VaR area at the final value of the VaR path
        var_final_value = simulated_paths[var_idx, -1]
        plt.fill_between(
            dates[idx_1y_start:],
            0,
            var_final_value,
            color=var_area_color,
            alpha=0.3,
            zorder=5,
            label='VaR Area'
        )
        # Plot the expected value (mean path)
        mean_path = np.mean(simulated_paths, axis=0)
        plt.plot(dates[idx_1y_start:], mean_path[1:], color='tab:red', alpha=0.7, linewidth=2, label='Expected Value (Mean Path)')
        # Plot actual last year prices
        plt.plot(dates[idx_1y_start:], open_prices[idx_1y_start:], label='Historical Last Year', color='tab:blue')
        # Extract stock name from file path (e.g., 'AAPL' from 'AAPL_daily_5y.csv')
        stock_name = os.path.basename(file).split('_')[0].upper()
        plt.title(f'{stock_name}: Historical Last Year vs Simulated GBM Paths')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        # Make x-axis ticks sparse and format dates
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[::21])  # Show roughly every month (21 trading days)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        plt.tight_layout()
        # Save the plot instead of showing it
        output_dir = os.path.join(os.path.dirname(__file__), '../tests')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{stock_name}_gbm_simulation_plot.png')
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


    #metrics
    actual_returns = paths_to_returns(open_prices[idx_1y_start:].reshape(1, -1))
    simulated_returns = paths_to_returns(simulated_paths)
    
    print(f"Actual last year return: {actual_returns.mean():.4f}")
    print(f"Simulated last year return: {simulated_returns.mean():.4f}")
    print(f"Simulated last year VaR (95%): {value_at_risk(simulated_returns, confidence_level=0.95):.4f}")

    print("difference in mean return:", abs(actual_returns.mean() - simulated_returns.mean()))

for file in ['../data/AAPL_daily_5y.csv', '../data/MSFT_daily_5y.csv', '../data/GOOGL_daily_5y.csv', '../data/MSFT_daily_5y.csv', '../data/TSLA_daily_5y.csv']:
    print(f"\nComparing historical data to GBM simulation for {file}...")
    compare_hist_to_gbm(file, num_sims=1000)