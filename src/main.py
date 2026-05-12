"""
main.py — end-to-end portfolio simulation pipeline.

Usage (from repo root):
    python src/main.py
"""

import sys
import glob
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils.util import calibrate_portfolio_from_csvs, calibrate_from_csv
from simulations.sim import simulate_portfolio, plot_portfolio_distribution

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PATHS       = 5_000
SEED        = 42
CONF        = 0.95
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_daily_5y.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    # 1. Calibrate per-equity GBM parameters from historical data
    print("Calibrating GBM parameters from CSVs...")
    assets = calibrate_portfolio_from_csvs(csv_files)

    ticker_names = [os.path.basename(f).split("_")[0].upper() for f in csv_files]
    print(f"\n{'Ticker':<8} {'mu':>8} {'sigma':>8} {'weight':>8}")
    print("-" * 36)
    for name, (mu, sigma, w) in zip(ticker_names, assets):
        # also pull S0 for display
        _, _, S0 = calibrate_from_csv(
            os.path.join(DATA_DIR, f"{name}_daily_5y.csv")
        )
        print(f"{name:<8} {mu:>8.4f} {sigma:>8.4f} {w:>8.4f}  S0={S0:.2f}")

    # 2. Run portfolio simulation
    print(f"\nRunning {PATHS:,} Monte-Carlo paths...")
    port_returns = simulate_portfolio(assets, paths=PATHS, seed=SEED)

    # 3. Print summary statistics
    var  = -np.percentile(port_returns, (1 - CONF) * 100)
    tail = port_returns[port_returns <= np.percentile(port_returns, (1 - CONF) * 100)]
    cvar = -float(tail.mean())

    print(f"\n── Portfolio return summary ({PATHS:,} paths) ──────────────────")
    print(f"  Mean return : {port_returns.mean():.4f}")
    print(f"  Std  return : {port_returns.std():.4f}")
    print(f"  VaR  {int(CONF*100)}%    : {var:.4f}")
    print(f"  CVaR {int(CONF*100)}%    : {cvar:.4f}")
    print(f"  Min  return : {port_returns.min():.4f}")
    print(f"  Max  return : {port_returns.max():.4f}")

    # 4. Plot and save
    tickers_str = " / ".join(ticker_names)
    save_path   = os.path.join(RESULTS_DIR, "portfolio_distribution.png")
    plot_portfolio_distribution(
        port_returns,
        title=f"Portfolio Return Distribution ({tickers_str})",
        confidence_level=CONF,
        save_path=save_path,
        show=True,
    )


if __name__ == "__main__":
    main()
