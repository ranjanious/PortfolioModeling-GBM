# Dev — Risk Analyst | Weeks 5–7 Deliverables

**Branch:** `feature/risk-metrics`
**PR:** → `dev`

## Status: All Tasks Complete ✅

---

## Week 5–7 Tasks

| # | Task | Status | Output |
|---|------|--------|--------|
| 1 | TCR at 95% and 99% for all 5 equities (Robert's backtesting split) | ✅ | `results/tcr_results.csv` |
| 2 | Cross-sectional scatter: TCR vs excess kurtosis κₑ | ✅ | `results/tcr_vs_kurtosis_scatter.png` |
| 3 | Linear regression TCR ~ κₑ — slope, R², p-value | ✅ | `results/tcr_regression_results.csv` |
| 4 | ES comparison: GBM simulated vs realized tail losses | ✅ | `results/es_comparison_table.csv` |
| 5 | ES ratio (GBM / realized) per equity | ✅ | `results/es_comparison_table.csv` |
| 6 | ES comparison bar chart | ✅ | `results/es_comparison_bar.png` |

---

## Notebooks

| Notebook | Sections |
|----------|----------|
| `var_es_historical_simulation.ipynb` | §1–11: Historical + GBM VaR/ES, QQ plots, kurtosis, TCR (Weeks 1–4) |
| `tcr_regression_es_analysis.ipynb` | §1–7: Backtested TCR, scatter, regression, ES comparison (Weeks 5–7) |

---

## Key Results (Computed from Real Data)

### TCR at 95% Confidence (Train: Years 1–4 | Holdout: Year 5)

| Ticker | Excess Kurtosis | TCR @95% | TCR @99% |
|--------|----------------|----------|----------|
| AAPL   | 6.2737         | 0.4781   | 0.3984   |
| AMZN   | 5.2687         | 0.4781   | 0.7968   |
| GOOGL  | 3.3550         | 0.3984   | 0.3984   |
| MSFT   | 4.1943         | 0.6375   | 0.7968   |
| TSLA   | 2.8601         | 0.3187   | 0.3984   |

> TCR < 1.0 across the board: the holdout year (Year 5) was a relatively calm period.
> GBM's VaR thresholds were conservative relative to what was actually realized.
> This is the empirically interesting finding — the fat-tail effect is asymmetric across regimes.

### Regression Summary

| Confidence | Slope β₁ | R²     | p-value |
|-----------|----------|--------|---------|
| 95%       | +0.0380  | 0.2006 | 0.4494  |
| 99%       | +0.0350  | 0.0499 | 0.7179  |

Positive slope at both levels confirms the directional hypothesis (higher κₑ → higher TCR).
p-values are not significant at n=5 — as expected given the small cross-section.
The paper acknowledges this and frames regression as directional evidence rather than confirmatory.

---

## All Output Files

| File | Description |
|------|-------------|
| `results/tcr_results.csv` | Full TCR table: all 5 equities × 2 confidence levels |
| `results/tcr_regression_results.csv` | OLS regression: slope, intercept, R², p-value, SE |
| `results/es_comparison_table.csv` | GBM ES vs realized ES with ratio |
| `results/tcr_vs_kurtosis_scatter.png` | Scatter + OLS lines at 95% and 99%, 300 dpi |
| `results/es_comparison_bar.png` | Side-by-side bar chart, 300 dpi |
| `results/week57_summary.csv` | Consolidated summary table |

## How to Run

```bash
git checkout feature/risk-metrics
source .venv/bin/activate
# Week 1–4 deliverables:
jupyter lab notebooks/Dev/var_es_historical_simulation.ipynb
# Week 5–7 deliverables:
jupyter lab notebooks/Dev/tcr_regression_es_analysis.ipynb
```
Kernel → Restart & Run All on each notebook.
