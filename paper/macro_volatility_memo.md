# MEMORANDUM

**TO:** Ayush (Lead Researcher)  
**FROM:** Nihan (Data Engineer)  
**DATE:** June 5, 2026  
**SUBJECT:** FRED Macroeconomic Series Correlations with Realized Volatility (For Section 7: Limitations and Failure Modes)

---

### 1. Executive Summary
This memo summarizes the empirical correlation analysis between three Federal Reserve Economic Data (FRED) macroeconomic indicators—**Consumer Price Index (CPIAUCSL)**, **10-Year Treasury Yield (DGS10)**, and the **Unemployment Rate (UNRATE)**—and the 30-day realized volatility of our 30-stock expanded equity universe over the 5-year sample period (2018–2026). 

The empirical findings demonstrate that realized equity volatility is not a constant, isolated parameter as assumed under the standard Geometric Brownian Motion (GBM) framework. Instead, it is statistically dependent on broader macroeconomic cycles, particularly the labor market and interest rate regimes. These results provide direct empirical justification for Section 7, outlining why global GBM models systematically underestimate tail risks (VaR/ES) during macro-regime shifts.

---

### 2. Empirical Findings & Correlation Analysis
Using daily business-day aligned returns, we computed the 30-day rolling standard deviation for each of the 30 equities and aligned them with the contemporaneous values of the FRED macro series. The full correlation table is saved in [macro_volatility_correlation.csv](../results/macro_volatility_correlation.csv). Key relationships include:

#### A. The Labor Market Cycle (UNRATE)
The Unemployment Rate exhibits a strong, positive correlation with realized equity volatility across the entire universe, averaging **0.42** across all 30 equities. This relationship is most pronounced in cyclical and systemically sensitive sectors:
*   **Financials:** `BAC` (0.6229), `JPM` (0.6101), `GS` (0.5621)
*   **Energy:** `XOM` (0.6577), `COP` (0.6061), `CVX` (0.6216)
*   **Industrials:** `BA` (0.6814), `HON` (0.6262)

*Interpretation:* Rising unemployment acts as a proxy for macroeconomic contractions and recessionary stress. During these regimes, market uncertainty spikes, leading to volatility clustering.

#### B. The Interest Rate Environment (DGS10)
The 10-Year Treasury Yield displays a persistent negative correlation with realized volatility (averaging **-0.21**). This correlation is strongest in debt-dependent and cyclical sectors:
*   **Energy/Industrials:** `COP` (-0.4459), `BA` (-0.3949), `BAC` (-0.3813)
*   **Technology:** `AAPL` (-0.2822), `MSFT` (-0.2095)

*Interpretation:* In the current sample period, falling bond yields (often associated with flights to safety or monetary easing cycles) correspond with periods of high equity market stress and elevated volatility.

#### C. Price Inflation (CPIAUCSL)
Inflation displays a mixed and generally weak correlation with realized volatility, showing mild negative relationships for cyclical tech (`AAPL`: -0.2349) but positive relationships for defensive and inflation-hedged sectors (e.g., Real Estate/Utilities/Healthcare: `NKE` at 0.1945, `UNH` at 0.1412, `NEE` at 0.1353).

---

### 3. Implications for Section 7 (Limitations of GBM)
These empirical results identify two critical failure modes of the standard GBM model that must be addressed in the paper:

1.  **Violation of Constant Volatility ($\sigma$):** 
    GBM assumes that the diffusion coefficient $\sigma$ is constant. However, our results demonstrate that volatility fluctuates dynamically and is statistically tied to external macroeconomic variables. When macro regimes shift (e.g., a rapid increase in unemployment), realized volatility rises rapidly, rendering a globally calibrated $\sigma$ obsolete.
2.  **Volatility Clustering & Fat Tails:** 
    Because volatility is conditioned on persistent macro states (recessions, rate cycles), volatility displays high autocorrelation (clustering). Under constant-parameter GBM, the probability of extreme returns is governed by a normal distribution, which decays exponentially. In reality, macro-conditioned volatility clustering produces heavy tails. As a result, globally calibrated GBM models systematically underestimate the frequency and magnitude of tail breaches, leading to a **Tail Coverage Ratio (TCR)** significantly greater than 1.0.

### 4. Recommendations for Section 7 Drafting
To illustrate these limitations, Section 7 should incorporate:
*   A discussion of the **UNRATE-volatility link** as evidence that equity risks are tied to the business cycle.
*   The **sub-period correlation heatmaps** showing that cross-asset correlations shift toward 1.0 during macro shocks (like March 2020), violating the independent-increment assumption.
*   A justification for **Robert's regime-conditioned calibration extension** (separating parameters into High/Low volatility states using your rolling 75th percentile thresholds) as a direct method to mitigate these tail risk underestimation errors.
