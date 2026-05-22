from models.gbm import GBM
from utils.util import (
    value_at_risk,
    paths_to_returns,
    extract_daily_open,
    csv_to_numpy_with_dates,
    calibrate_from_csv,
    calibrate_portfolio_from_csvs,
)
import matplotlib.pyplot as plt
import numpy as np
import os


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    N: int = 252,
    paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Run a GBM simulation and return simulated price paths.

    Args:
        S0 (float): Initial asset price.
        mu (float): Annualised drift.
        sigma (float): Annualised volatility.
        T (float): Time horizon in years. Defaults to 1.0.
        N (int): Number of time steps. Defaults to 252 (trading days).
        paths (int): Number of Monte-Carlo paths. Defaults to 1000.
        seed (int | None): RNG seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Shape (paths, N+1) array of simulated price paths.
    """
    gbm = GBM(S0=S0, mu=mu, sigma=sigma, T=T, N=N, seed=seed)
    return gbm.simulate(paths=paths, show=False)


def simulate_portfolio(
    assets: list[tuple[float, float, float]],
    S0: float = 100.0,
    T: float = 1.0,
    N: int = 252,
    paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a weighted portfolio of GBM assets and return the portfolio
    return distribution across all Monte-Carlo paths.

    Each asset is simulated independently via simulate_gbm(), its per-path
    returns are computed, and the weighted sum gives the portfolio return for
    each path.

    Args:
        assets (list[tuple[float, float, float]]): A list of (mu, sigma, weight)
            tuples — one per asset.  Weights need not sum to 1; they are
            normalised internally.
        S0 (float): Common initial price for all assets. Defaults to 100.0.
        T (float): Time horizon in years. Defaults to 1.0.
        N (int): Number of time steps. Defaults to 252.
        paths (int): Number of Monte-Carlo paths. Defaults to 1000.
        seed (int | None): Base RNG seed.  Each asset receives a
            deterministically offset seed so paths are independent.

    Returns:
        np.ndarray: Shape (paths,) array of weighted portfolio returns.

    Raises:
        ValueError: If assets is empty or any weight is negative.
    """
    if not assets:
        raise ValueError("assets list must contain at least one (mu, sigma, weight) tuple.")

    mus, sigmas, weights = zip(*assets)
    weights = np.array(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")

    total_weight = weights.sum()
    if total_weight == 0:
        raise ValueError("Sum of weights must be greater than zero.")
    weights = weights / total_weight  # normalise

    portfolio_returns = np.zeros(paths)

    for i, (mu, sigma, w) in enumerate(zip(mus, sigmas, weights)):
        asset_seed = (seed + i) if seed is not None else None
        price_paths = simulate_gbm(
            S0=S0, mu=mu, sigma=sigma, T=T, N=N, paths=paths, seed=asset_seed
        )
        asset_returns = paths_to_returns(price_paths)  # shape (paths,)
        portfolio_returns += w * asset_returns

    return portfolio_returns


def plot_portfolio_distribution(
    portfolio_returns: np.ndarray,
    title: str = "Portfolio Return Distribution",
    confidence_level: float = 0.95,
    bins: int = 60,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a histogram of the simulated portfolio return distribution with
    VaR and CVaR overlays.

    Args:
        portfolio_returns (np.ndarray): Shape (paths,) array as produced by
            simulate_portfolio().
        title (str): Plot title.
        confidence_level (float): Confidence level for VaR / CVaR lines.
            Defaults to 0.95.
        bins (int): Number of histogram bins. Defaults to 60.
        save_path (str | None): If given, figure is saved here.
        show (bool): Whether to call plt.show(). Defaults to True.
    """
    var = value_at_risk(portfolio_returns, confidence_level=confidence_level)
    tail_mask = portfolio_returns <= np.percentile(
        portfolio_returns, (1 - confidence_level) * 100
    )
    cvar = -float(portfolio_returns[tail_mask].mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(portfolio_returns, bins=bins, color="steelblue", edgecolor="white",
            alpha=0.85, label="Simulated returns")
    ax.axvline(-var, color="crimson", linewidth=2,
               label=f"VaR {int(confidence_level * 100)}%  = {var:.2%}")
    ax.axvline(-cvar, color="darkorange", linewidth=2, linestyle="--",
               label=f"CVaR {int(confidence_level * 100)}% = {cvar:.2%}")
    ax.axvline(float(portfolio_returns.mean()), color="limegreen", linewidth=2,
               linestyle=":", label=f"Mean return = {portfolio_returns.mean():.2%}")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


