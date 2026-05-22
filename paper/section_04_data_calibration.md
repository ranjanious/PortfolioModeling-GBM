# Section 4: Data and Calibration Methodology

**Draft status:** Complete — Week 2 sprint (Dev)
**Branch:** `feature/risk-metrics`

---

## 4.1 Overview

This section describes the empirical data pipeline that underpins all
downstream risk calculations in the paper. Historical equity price data
is obtained through the `yfinance` library, supplemented by macroeconomic
series from the Federal Reserve Economic Data (FRED) API. Geometric
Brownian Motion (GBM) parameters—annualised drift $\hat{\mu}$ and
volatility $\hat{\sigma}$—are then estimated from these historical records
and used both to drive the Monte Carlo simulation engine (Sections 5–6)
and to benchmark against non-parametric historical-simulation risk
metrics (Section 6).

---

## 4.2 Equity Data — yFinance Pipeline

### 4.2.1 Source and Coverage

Daily open, high, low, closing, and adjusted-closing prices, together
with traded volume, were retrieved via the `yfinance` Python library
(v0.2.40) for five large-capitalisation U.S. equities. Adjusted closing
prices incorporate dividends and split adjustments and are the primary
series used for log-return computation.

**Equity Universe**

| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Technology |
| AMZN | Amazon.com Inc. | Consumer Discretionary |
| GOOGL | Alphabet Inc. (Class A) | Communication Services |
| MSFT | Microsoft Corporation | Technology |
| TSLA | Tesla Inc. | Consumer Discretionary |

These five stocks represent a cross-section of the largest U.S.
technology and growth equities, deliberately chosen to test whether
GBM tail-risk underestimation is a consistent phenomenon across high-
volatility, high-kurtosis assets or an idiosyncratic effect.

### 4.2.2 Sample Period

Each CSV covers approximately five years of daily observations,
beginning April 2021. The full window spans through late 2026,
yielding approximately 1,250 trading days per equity. All five series
share the same calendar, permitting direct cross-asset comparisons and
the construction of a $5 \times 5$ cross-asset correlation matrix
(Section 4.4).

### 4.2.3 Data Cleaning

The raw downloads are validated using the `notebooks/validate_equities.ipynb`
notebook (Nihan), which performs the following checks before any
calibration or simulation:

1. **Date range confirmation** — verify each CSV spans the expected
   window with no silent truncation.
2. **Column name validation** — assert presence of `Date`, `Open`,
   `High`, `Low`, `Close`, `Adj Close`, `Volume`.
3. **Missing value audit** — flag any `NaN` entries; in practice, the
   `yfinance` pull returns a complete panel for these five liquid names
   with no missing observations.
4. **Summary statistics table** — mean, standard deviation, minimum,
   and maximum of daily adjusted-closing prices and log-returns.

Only after passing all four checks are the CSVs admitted into the
calibration and simulation pipeline.

---

## 4.3 Macroeconomic Data — FRED API

Macroeconomic context series were retrieved from the St. Louis Federal
Reserve's FRED API using the `fredapi` Python library (v0.5.2). Three
series are stored in `/data/`:

| FRED Code | Series | Frequency |
|-----------|--------|-----------|
| `DGS10` | 10-Year U.S. Treasury Constant Maturity Rate | Daily |
| `CPIAUCSL` | Consumer Price Index for All Urban Consumers | Monthly |
| `UNRATE` | Civilian Unemployment Rate | Monthly |

At the modelling stage used in this paper, the FRED series serve as
contextual descriptors of the macroeconomic regime rather than
regressors in the GBM calibration. A natural extension would be to
allow $\hat{\mu}$ to vary with the yield environment via a
regime-switching layer; this is noted as future work in Section 8.

---

## 4.4 Cross-Asset Correlation Matrix

The $5 \times 5$ cross-asset correlation matrix of daily log-returns
is computed as the sample Pearson correlation matrix:

$$
\Sigma_{ij} = \frac{\sum_{t=1}^{T}(r_{i,t} - \bar{r}_i)(r_{j,t} - \bar{r}_j)}
                   {(T-1)\,\hat{\sigma}_i\,\hat{\sigma}_j}
$$

where $r_{i,t}$ denotes the daily log-return of equity $i$ on day $t$
and $T$ is the common sample length. The matrix is saved to
`/results/correlation_matrix_5x5.csv`. Notable features of the
empirical correlation structure—such as the elevated AAPL–MSFT pairwise
correlation and the lower correlations involving TSLA—are discussed in
Section 5 in the context of portfolio diversification effects.

---

## 4.5 GBM Parameter Calibration

### 4.5.1 Estimation Procedure

The GBM model posits that the log-price $\ln S_t$ follows a Brownian
motion with constant drift and diffusion. Under the discrete daily
approximation, the log-return at each step is:

$$
r_t = \ln\!\left(\frac{S_t}{S_{t-1}}\right) \sim
\mathcal{N}\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t,\;
\sigma^2 \Delta t\right]
$$

where $\Delta t = 1/252$ (one trading day). Given a sample of $T$
observed daily log-returns $\{r_1, \ldots, r_T\}$, we estimate the
GBM parameters as follows.

**Step 1 — Sample moments of daily log-returns:**

$$
\bar{r} = \frac{1}{T}\sum_{t=1}^{T} r_t, \qquad
s_r = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar{r})^2}
$$

**Step 2 — Annualise:**

$$
\hat{\sigma} = s_r \times \sqrt{252}, \qquad
\hat{\mu}_{\text{arithmetic}} = \bar{r} \times 252
$$

**Step 3 — Itô correction to recover the GBM drift parameter:**

The sample mean $\bar{r}$ estimates the *log-return drift*
$(\mu - \sigma^2/2)$, not $\mu$ itself. Solving for the GBM
drift parameter gives:

$$
\hat{\mu} = \hat{\mu}_{\text{arithmetic}} + \frac{\hat{\sigma}^2}{2}
$$

This correction ensures that $e^{\mu T}$ equals the expected value of
the price ratio $S_T / S_0$ under the calibrated model, consistent
with the GBM closed-form solution.

### 4.5.2 Calibration Window

The calibration uses the most-recent four years of daily data,
discarding the most-recent year for out-of-sample validation of the
risk metrics. This convention—implemented in
`src/utils/util.py::calibrate_from_csv(use_years=4.0)`—is applied
consistently across all five equities.

### 4.5.3 Calibrated Parameter Summary

The table below reports the estimated parameters for each equity, as
produced by the calibration script in `src/simulations/sim.py` and
cross-checked within this notebook.

| Ticker | $\hat{\mu}$ (ann.) | $\hat{\sigma}$ (ann.) | $S_0$ (last open, USD) |
|--------|--------------------|-----------------------|------------------------|
| AAPL | calibrated from CSV | calibrated from CSV | last open price |
| AMZN | calibrated from CSV | calibrated from CSV | last open price |
| GOOGL | calibrated from CSV | calibrated from CSV | last open price |
| MSFT | calibrated from CSV | calibrated from CSV | last open price |
| TSLA | calibrated from CSV | calibrated from CSV | last open price |

*Exact values are printed by the calibration cells in
`notebooks/Dev/var_es_historical_simulation.ipynb` Section 5, which
reads directly from the CSVs at runtime. Values are intentionally not
hard-coded here to maintain reproducibility as the data pipeline is
updated.*

### 4.5.4 Limitations of the Calibration

Two important caveats apply to the parameter estimates:

1. **Stationarity assumption.** The calibration implicitly assumes that
   $\mu$ and $\sigma$ are constant across the full sample window.
   In reality, equity volatility is time-varying (volatility clustering),
   and the mean return is highly sensitive to the choice of window.
   A GARCH(1,1) calibration would partially address the first issue;
   this is noted as future work.

2. **Normal distribution assumption.** As evidenced by the excess
   kurtosis table in Section 9 of the notebook, all five equities
   exhibit positive excess kurtosis (leptokurtosis), meaning their
   empirical return distributions have heavier tails than the normal
   distribution implied by GBM. This structural mismatch is the root
   cause of GBM's systematic VaR underestimation at the 99% confidence
   level, quantified by the Tail Coverage Ratio (TCR) in Section 10.

---

## 4.6 Data and Code Reproducibility

All data retrieval, cleaning, and calibration steps are fully scripted
and version-controlled:

| Script / Notebook | Responsibility | Branch |
|-------------------|---------------|--------|
| `src/yfinance_pull.py` | Pull equity CSVs | `feature/data-pipeline` |
| `src/fred_pull.py` | Pull FRED series | `feature/data-pipeline` |
| `notebooks/validate_equities.ipynb` | Validation & summary stats | `feature/data-pipeline` |
| `src/utils/util.py::calibrate_from_csv()` | GBM calibration | `feature/simulation` |
| `notebooks/Dev/var_es_historical_simulation.ipynb` | Risk metric computation | `feature/risk-metrics` |

To reproduce all results from a fresh clone:

```bash
git clone https://github.com/ranjanious/PortfolioModeling-GBM.git
cd PortfolioModeling-GBM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Run data pipeline (requires yfinance / FRED API access)
python src/yfinance_pull.py
python src/fred_pull.py
# Run risk metrics notebook
jupyter lab notebooks/Dev/var_es_historical_simulation.ipynb
```

All figures are saved automatically to `/results/` and `/results/qq_plots/`
at 300 dpi. The simulation and calibration modules in `/src/` are
importable with no additional setup after `pip install -r requirements.txt`.

---

*References for this section:*
- Nihan's data pipeline: `feature/data-pipeline` branch, `src/yfinance_pull.py`, `src/fred_pull.py`
- Robert's calibration module: `feature/simulation` branch, `src/utils/util.py`
- Hull, J. C. (2018). *Risk Management and Financial Institutions* (5th ed.). Wiley. Chapter 15.
