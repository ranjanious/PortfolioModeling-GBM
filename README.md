# PortfolioModeling-GBM

Monte Carlo portfolio modeling with Geometric Brownian Motion (GBM), calibrated
on equity price data from `yfinance` and macro data from FRED. The project
produces simulation outputs, risk metrics (VaR, ES, drawdown, etc.), diagnostic
plots (QQ-plots, fan charts), and an accompanying academic paper.

## Repository Structure

| Folder / File      | Purpose                                                                       |
|--------------------|-------------------------------------------------------------------------------|
| `/data/`           | Raw and cleaned equity price data pulled from yfinance and FRED               |
| `/notebooks/`      | Jupyter notebooks for simulation, calibration, and risk metric development    |
| `/src/`            | Production-grade Python modules for the Monte Carlo engine and risk calculations |
| `/paper/`          | LaTeX or Word drafts, bibliography files, and figure exports                  |
| `/results/`        | Saved charts, QQ-plots, simulation outputs, and risk metric tables            |
| `README.md`        | Project overview, setup instructions, and contribution guidelines             |
| `requirements.txt` | Pinned Python dependencies for reproducibility across all team machines       |

## Setup

```bash
git clone https://github.com/ranjanious/portfoliomodeling-gbm.git
cd portfoliomodeling-gbm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Launch notebooks with:

```bash
jupyter lab
```

## Data Pull Scripts

Create a local `.env` file (do not commit it):

```bash
cp .env.example .env
# then set FRED_API_KEY in .env
```

Run yfinance pull:

```bash
python src/yfinance_pull.py
```

Run FRED pull (default: UNRATE, CPIAUCSL, DGS10, gap-filled to business days):

```bash
python src/fred_pull.py
```

## Branching Convention

| Branch                   | Owner                  | Purpose                                              |
|--------------------------|------------------------|------------------------------------------------------|
| `main`                   | Team (review required) | Stable, reviewed code and paper drafts only; no direct commits |
| `dev`                    | Team                   | Integration branch; all feature branches merge here first      |
| `feature/data-pipeline`  | Data Engineer          | yfinance / FRED data ingestion and cleaning          |
| `feature/simulation`     | Simulation Developer   | Monte Carlo engine, GBM calibration                  |
| `feature/risk-metrics`   | Risk Analyst           | VaR, ES, drawdown, and other risk calculations       |
| `feature/paper`          | Writer / Statistician  | LaTeX draft, figures, bibliography                   |

### Workflow

1. Create or check out your feature branch from `dev`.
2. Commit work with clear, descriptive messages.
3. Open a pull request into `dev` for review.
4. `dev` is merged into `main` only after review and milestone sign-off.

## Contribution Guidelines

- Never commit directly to `main`.
- Keep feature branches focused on a single concern.
- Run notebooks top-to-bottom and clear outputs before committing.
- Pin any new dependency in `requirements.txt`.
- Large data files belong in `/data/` and should be git-ignored if above
  repository size limits; document the retrieval script instead.
