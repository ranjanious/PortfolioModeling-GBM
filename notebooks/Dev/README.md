# Dev — Risk Analyst | `feature/risk-metrics`

## Week 1–2 Deliverables

| # | Deliverable | Status | File |
|---|-------------|--------|------|
| 1 | Historical simulation VaR at 95% and 99% | ✅ | `var_es_historical_simulation.ipynb` |
| 2 | Expected Shortfall at 95% and 99% | ✅ | `var_es_historical_simulation.ipynb` |
| 3 | Summary table (VaR + ES, both confidence levels) | ✅ | Cell 6 of notebook |
| 4 | Return distribution histogram with VaR lines | ✅ | `var_es_historical_simulation.ipynb` |
| 5 | Figure saved to `/results/` at 300 dpi | ✅ | `AAPL_log_return_distribution_VaR_ES.png` |

> **Note:** Saurish is no longer on the team. Dev has absorbed the return distribution visualisation task (originally Saurish's) in addition to the VaR/ES prototype work.

---

## Notebook: `var_es_historical_simulation.ipynb`

### What it does
1. **Loads** closing price data — reads Nihan's CSV from `/data/` if present, falls back to `yfinance` download so it always runs from a fresh kernel.
2. **Computes** daily log-returns: $r_t = \ln(S_t / S_{t-1})$.
3. **Computes VaR** at 95% and 99% via empirical quantile of the sorted return distribution.
4. **Computes ES** at 95% and 99% as the conditional mean of returns below each VaR threshold.
5. **Plots** a histogram of log-returns with VaR vertical lines and a normal overlay, saved to `/results/`.
6. **Prints** a clean summary table and written interpretation for the methodology section.

### How to run
```bash
# From repo root, with .venv activated
jupyter lab notebooks/Dev/var_es_historical_simulation.ipynb
```
Or run all cells: `Kernel → Restart & Run All`

### Dependencies
All from `requirements.txt` — no additional installs needed:
- `numpy`, `pandas`, `scipy`, `matplotlib`

---

## Commit convention
```
[risk] <description>
```
Example: `[risk] add historical simulation VaR ES notebook`

## Branch
`feature/risk-metrics` — merge into `dev` via PR with one approval from Ayush.
