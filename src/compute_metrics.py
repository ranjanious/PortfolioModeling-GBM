from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from yfinance_pull import EQUITY_UNIVERSE


# ── Default ticker list — all 30 equities in the expanded universe ───────────
DEFAULT_TICKERS = list(EQUITY_UNIVERSE.keys())
PRICE_COLUMNS   = ["Adj Close", "Close"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-ticker log returns, 30-day rolling volatility, and the "
            "NxN cross-asset correlation matrix for all equities in the universe."
        )
    )
    parser.add_argument("--data-dir",   type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--tickers",    nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--window",     type=int,  default=30)
    parser.add_argument("--min-periods",type=int,  default=30)
    return parser.parse_args(argv)


def load_equity_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column missing in {path.name}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    for col in PRICE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def select_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    for col in PRICE_COLUMNS:
        if col in df.columns and not df[col].dropna().empty:
            if col != "Adj Close":
                print(f"Warning: {ticker} using '{col}' (no usable 'Adj Close').")
            s = df.set_index("Date")[col].astype(float)
            return s
    raise ValueError(f"{ticker} has no usable price column.")


def compute_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def compute_rolling_volatility(lr: pd.Series, window: int, min_periods: int) -> pd.Series:
    return lr.rolling(window=window, min_periods=min_periods).std()


def compute_summary_stats(lr: pd.Series, ticker: str) -> dict:
    return {
        "Ticker":           ticker,
        "Sector":           EQUITY_UNIVERSE.get(ticker, "Unknown"),
        "Obs":              int(lr.dropna().__len__()),
        "Ann_Return_pct":   round(float(lr.mean() * 252) * 100, 2),
        "Ann_Vol_pct":      round(float(lr.std()  * np.sqrt(252)) * 100, 2),
        "Daily_Mean":       round(float(lr.mean()), 6),
        "Daily_Std":        round(float(lr.std()),  6),
        "Skewness":         round(float(lr.skew()), 4),
        "Excess_Kurtosis":  round(float(lr.kurt()), 4),
        "Min_Return_pct":   round(float(lr.min()) * 100, 3),
        "Max_Return_pct":   round(float(lr.max()) * 100, 3),
    }


def main() -> None:
    args       = parse_args([])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing metrics for {len(args.tickers)} equities...")
    all_lr: dict[str, pd.Series] = {}
    summary_rows = []
    failed = []

    for ticker in args.tickers:
        safe_ticker = ticker.replace("-", "_")
        csv_path = args.data_dir / f"{ticker}_daily_5y.csv"
        # Also try with underscore (BRK-B stored as BRK-B or BRK_B)
        if not csv_path.exists():
            csv_path = args.data_dir / f"{safe_ticker}_daily_5y.csv"
        try:
            df     = load_equity_csv(csv_path)
            prices = select_price_series(df, ticker)
            lr     = compute_log_returns(prices).dropna()
            vol    = compute_rolling_volatility(lr, args.window, args.min_periods)

            all_lr[ticker] = lr

            # Save log returns
            lr_out = pd.DataFrame({"Date": lr.index, "Log_Return": lr.values})
            lr_out.to_csv(output_dir / f"{safe_ticker}_log_returns.csv", index=False)

            # Save rolling volatility
            vol_out = pd.DataFrame({"Date": vol.index, "Rolling_Vol_30d": vol.values})
            vol_out.to_csv(output_dir / f"{safe_ticker}_rolling_volatility_30d.csv", index=False)

            summary_rows.append(compute_summary_stats(lr, ticker))
            print(f"  ✓ {ticker}")
        except Exception as exc:
            failed.append(ticker)
            print(f"  ✗ {ticker}: {exc}")

    # Cross-asset correlation matrix (NxN, aligned on common dates)
    if len(all_lr) > 1:
        lr_df = pd.DataFrame(all_lr).dropna()
        corr_matrix = lr_df.corr()
        n = len(all_lr)
        corr_matrix.to_csv(output_dir / f"correlation_matrix_{n}x{n}.csv")
        print(f"\n  Saved {n}×{n} correlation matrix → correlation_matrix_{n}x{n}.csv")

    # Summary statistics table
    if summary_rows:
        stats_df = pd.DataFrame(summary_rows)
        stats_df.to_csv(output_dir / "summary_stats_all_equities.csv", index=False)
        print(f"  Saved summary stats → summary_stats_all_equities.csv")

    if failed:
        print(f"\nFailed tickers: {failed}")
    else:
        print(f"\nAll {len(args.tickers)} equities processed successfully.")


if __name__ == "__main__":
    main()
