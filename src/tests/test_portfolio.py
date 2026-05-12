"""
Pytest tests for simulate_portfolio() in simulations/sim.py.

Covers:
  - output shape
  - internal weight normalisation (arbitrary raw weights → sum-to-1 behaviour)
  - equal-weight result matches manual weighted sum
  - mean-return convergence toward the theoretical expectation (LLN)
  - guard-rail error paths (empty assets, negative weights, zero weights)
"""

import sys
import os

# Allow running from the repo root or from src/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from simulations.sim import simulate_portfolio, simulate_gbm
from utils.util import paths_to_returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42  # must be non-zero: GBM.__init__ uses `seed if seed` which treats 0 as falsy

# Simple 2-asset portfolio: moderate drift assets
ASSETS_2 = [
    (0.10, 0.20, 0.5),   # mu=10%, sigma=20%, weight=50%
    (0.15, 0.25, 0.5),   # mu=15%, sigma=25%, weight=50%
]

# Single-asset portfolio (edge case)
ASSETS_1 = [(0.12, 0.18, 1.0)]

# 5-asset equal-weight portfolio
ASSETS_5 = [
    (0.10, 0.20, 0.2),
    (0.12, 0.22, 0.2),
    (0.08, 0.18, 0.2),
    (0.15, 0.30, 0.2),
    (0.05, 0.15, 0.2),
]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_default_paths(self):
        result = simulate_portfolio(ASSETS_2, paths=500, seed=SEED)
        assert result.shape == (500,), "Output must be 1-D with length == paths"

    def test_custom_paths(self):
        for n in (1, 100, 2000):
            result = simulate_portfolio(ASSETS_2, paths=n, seed=SEED)
            assert result.shape == (n,)

    def test_single_asset(self):
        result = simulate_portfolio(ASSETS_1, paths=200, seed=SEED)
        assert result.shape == (200,)

    def test_five_assets(self):
        result = simulate_portfolio(ASSETS_5, paths=300, seed=SEED)
        assert result.shape == (300,)

    def test_dtype_is_float(self):
        result = simulate_portfolio(ASSETS_2, paths=100, seed=SEED)
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# Weight normalisation tests
# ---------------------------------------------------------------------------

class TestWeightNormalisation:
    def test_unnormalised_weights_equal_normalised(self):
        """Passing weights [1, 1] and [0.5, 0.5] must yield identical results."""
        assets_raw = [(0.10, 0.20, 2.0), (0.15, 0.25, 2.0)]
        assets_norm = [(0.10, 0.20, 0.5), (0.15, 0.25, 0.5)]
        r_raw = simulate_portfolio(assets_raw, paths=500, seed=SEED)
        r_norm = simulate_portfolio(assets_norm, paths=500, seed=SEED)
        np.testing.assert_array_almost_equal(r_raw, r_norm, decimal=12)

    def test_large_weight_ratio(self):
        """Weights [100, 1] should behave like [0.99, 0.01] after normalisation."""
        assets = [(0.10, 0.20, 100.0), (0.15, 0.25, 1.0)]
        result = simulate_portfolio(assets, paths=200, seed=SEED)
        assert result.shape == (200,)

    def test_single_nonzero_weight_equals_single_asset(self):
        """Giving one asset a weight of 0 should match simulate_portfolio with only the live asset."""
        assets_full = [(0.10, 0.20, 1.0), (0.15, 0.25, 0.0)]
        assets_solo = [(0.10, 0.20, 1.0)]
        r_full = simulate_portfolio(assets_full, paths=500, seed=SEED)
        r_solo = simulate_portfolio(assets_solo, paths=500, seed=SEED)
        np.testing.assert_array_almost_equal(r_full, r_solo, decimal=12)


# ---------------------------------------------------------------------------
# Correctness: manual weighted-sum cross-check
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_equal_weight_matches_manual_sum(self):
        """For 2 equal-weight assets the portfolio return must equal the simple
        average of per-asset returns, computed independently with the same seeds."""
        paths = 800
        mu1, sigma1 = 0.10, 0.20
        mu2, sigma2 = 0.15, 0.25
        S0 = 100.0

        paths1 = simulate_gbm(S0=S0, mu=mu1, sigma=sigma1, paths=paths, seed=SEED)
        paths2 = simulate_gbm(S0=S0, mu=mu2, sigma=sigma2, paths=paths, seed=SEED + 1)
        manual = 0.5 * paths_to_returns(paths1) + 0.5 * paths_to_returns(paths2)

        assets = [(mu1, sigma1, 0.5), (mu2, sigma2, 0.5)]
        port = simulate_portfolio(assets, S0=S0, paths=paths, seed=SEED)

        np.testing.assert_array_almost_equal(port, manual, decimal=12)


# ---------------------------------------------------------------------------
# Mean-convergence (Law of Large Numbers)
# ---------------------------------------------------------------------------

class TestMeanConvergence:
    """With enough paths the sample mean should be close to the theoretical
    expected return.  GBM expected return over T=1 for a single asset is:
        E[R] = exp(mu * T) - 1  ≈  mu  for small mu
    For a portfolio: E[R_port] = sum_i w_i * (exp(mu_i) - 1).
    We use a loose tolerance because MC variance is intentional.
    """

    def _theoretical_mean(self, assets: list[tuple], T: float = 1.0) -> float:
        mus, _, weights = zip(*assets)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return float(np.sum(weights * (np.exp(np.array(mus) * T) - 1)))

    def test_single_asset_convergence(self):
        assets = [(0.10, 0.20, 1.0)]
        result = simulate_portfolio(assets, paths=20_000, seed=SEED)
        theoretical = self._theoretical_mean(assets)
        assert abs(result.mean() - theoretical) < 0.02, (
            f"Mean {result.mean():.4f} too far from theoretical {theoretical:.4f}"
        )

    def test_two_asset_convergence(self):
        assets = ASSETS_2
        result = simulate_portfolio(assets, paths=20_000, seed=SEED)
        theoretical = self._theoretical_mean(assets)
        assert abs(result.mean() - theoretical) < 0.02

    def test_five_asset_convergence(self):
        assets = ASSETS_5
        result = simulate_portfolio(assets, paths=20_000, seed=SEED)
        theoretical = self._theoretical_mean(assets)
        assert abs(result.mean() - theoretical) < 0.02

    def test_higher_mu_yields_higher_mean(self):
        """Portfolio with higher drift should have a higher mean return."""
        low_drift = [(0.05, 0.20, 1.0)]
        high_drift = [(0.25, 0.20, 1.0)]
        r_low = simulate_portfolio(low_drift, paths=5_000, seed=SEED)
        r_high = simulate_portfolio(high_drift, paths=5_000, seed=SEED)
        assert r_high.mean() > r_low.mean()


# ---------------------------------------------------------------------------
# Guard-rail / error-path tests
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_empty_assets_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            simulate_portfolio([], paths=100, seed=SEED)

    def test_negative_weight_raises(self):
        assets = [(0.10, 0.20, -0.5), (0.15, 0.25, 1.5)]
        with pytest.raises(ValueError, match="non-negative"):
            simulate_portfolio(assets, paths=100, seed=SEED)

    def test_all_zero_weights_raises(self):
        assets = [(0.10, 0.20, 0.0), (0.15, 0.25, 0.0)]
        with pytest.raises(ValueError, match="greater than zero"):
            simulate_portfolio(assets, paths=100, seed=SEED)
