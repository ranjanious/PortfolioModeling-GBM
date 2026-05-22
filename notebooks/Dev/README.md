# Dev — Risk Analyst | Week 2 Deliverables

**Branch:** `feature/risk-metrics`
**PR:** → `dev` (see pull request)

## Completed Tasks (Week 2 Sprint)

| # | Task | Status | Output |
|---|------|--------|--------|
| 1 | Rebase `feature/risk-metrics` onto `dev` | ✅ Done | Branch fast-forwarded to dev HEAD |
| 2 | Confirm notebook runs against Nihan's CSVs | ✅ Done | All 5 CSVs resolve; no fallback needed |
| 3 | Confirm notebook runs with Robert's GBM module | ✅ Done | `GBM` class used in §5 |
| 4 | Open PR into `dev` | ✅ Done | See PR |
| 5 | GBM VaR & ES alongside historical simulation | ✅ Done | `var_es_historical_simulation.ipynb` §5–6 |
| 6 | Direct comparison table (hist vs GBM) | ✅ Done | `var_es_historical_simulation.ipynb` §6 |
| 7 | QQ-plots for all 5 equities | ✅ Done | `results/qq_plots/qq_*.png` + grid |
| 8 | Excess kurtosis summary table | ✅ Done | `var_es_historical_simulation.ipynb` §9 |
| 9 | Tail Coverage Ratio (TCR) novelty metric | ✅ Done | `var_es_historical_simulation.ipynb` §10 |
| 10 | Section 4 paper draft (Data & Calibration) | ✅ Done | `paper/section_04_data_calibration.md` |

## Notebook Sections

| Section | Content |
|---------|---------|
| §0 | Imports & configuration |
| §1 | Data loading (all 5 Nihan CSVs) |
| §2 | Log-return definition with formula |
| §3 | Historical simulation VaR (95%, 99%) |
| §4 | Historical simulation ES (95%, 99%) |
| §5 | GBM-simulated VaR & ES (calibrated from data) |
| §6 | Full comparison table — all 5 equities × 2 confidence levels |
| §7 | Return distribution histograms with VaR lines (saved to `/results/`) |
| §8 | QQ-plots — individual + 5-panel grid (saved to `/results/qq_plots/`) |
| §9 | Excess kurtosis table — fat-tail diagnostic |
| §10 | Tail Coverage Ratio (TCR) — paper's novelty contribution |
| §11 | Reproducibility checklist |

## Key Outputs

| File | Description |
|------|-------------|
| `results/{TICKER}_log_return_distribution_VaR_ES.png` | Histogram + VaR lines, 300 dpi (×5) |
| `results/qq_plots/qq_{TICKER}.png` | Individual QQ plots, 300 dpi (×5) |
| `results/qq_all_equities.png` | 5-panel QQ grid |
| `paper/section_04_data_calibration.md` | Complete Section 4 draft |

## How to Run

```bash
git checkout feature/risk-metrics
source .venv/bin/activate
jupyter lab notebooks/Dev/var_es_historical_simulation.ipynb
# Kernel → Restart & Run All
```

All outputs reproduce from a fresh kernel with no errors.

## Notes on Saurish Handoff

Saurish's QQ-plot and distribution visualization scope was fully absorbed
into §7 and §8 of this notebook during Week 1. No tasks remain unassigned
from that handoff.
