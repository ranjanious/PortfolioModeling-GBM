from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
PRICE_COLUMNS = ["Adj Close", "Close"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-ticker log returns, 30-day rolling volatility, and "
            "a cross-asset correlation matrix from equity CSVs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing *_daily_5y.csv files. Default: ./data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Directory to write output CSVs. Default: ./results",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Space-separated tickers. Default: AAPL AMZN GOOGL MSFT TSLA",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size for volatility. Default: 30",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=30,
        help="Minimum periods for rolling volatility. Default: 30",
    )
    return parser.parse_args(argv)


def load_equity_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column missing in {path.name}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        raise ValueError(f"No valid dates in {path.name}")

    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    for col in PRICE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if "Adj Close" in df.columns and not df["Adj Close"].dropna().empty:
        series = df["Adj Close"]
    elif "Close" in df.columns and not df["Close"].dropna().empty:
        print(f"Warning: {ticker} using 'Close' (no usable 'Adj Close').")
        series = df["Close"]
    else:
        raise ValueError(f"{ticker} has no usable 'Adj Close' or 'Close' data.")

    series = series.astype(float)
    return series


def compute_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def compute_rolling_volatility(log_returns: pd.Series, window: int, min_periods: int) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=min_periods).std()


def save_series(series: pd.Series, date_series: pd.Series, output_path: Path, value_col: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"Date": date_series.dt.strftime("%Y-%m-%d"), value_col: series})
    out.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    if args.window <= 0:
        raise ValueError("--window must be > 0")
    if args.min_periods <= 0:
        raise ValueError("--min-periods must be > 0")
    if args.min_periods > args.window:
        raise ValueError("--min-periods cannot be greater than --window")

    tickers = [ticker.upper() for ticker in args.tickers]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned_returns: dict[str, pd.Series] = {}

    for ticker in tickers:
        csv_path = args.data_dir / f"{ticker}_daily_5y.csv"
        df = load_equity_csv(csv_path)

        if len(df) < args.min_periods:
            raise ValueError(
                f"{ticker} has insufficient rows ({len(df)}) for min periods {args.min_periods}."
            )

        price = select_price_series(df, ticker=ticker)
        log_returns = compute_log_returns(price)
        rolling_vol = compute_rolling_volatility(
            log_returns, window=args.window, min_periods=args.min_periods
        )

        aligned_returns[ticker] = pd.Series(log_returns.values, index=df["Date"], name=ticker)

        returns_path = args.output_dir / f"{ticker}_log_returns.csv"
        vol_path = args.output_dir / f"{ticker}_rolling_volatility_{args.window}d.csv"
        save_series(log_returns, df["Date"], returns_path, "log_return")
        save_series(rolling_vol, df["Date"], vol_path, f"rolling_volatility_{args.window}d")

        print(f"Saved {ticker} returns: {returns_path}")
        print(f"Saved {ticker} volatility: {vol_path}")

    returns_df = pd.DataFrame(aligned_returns).dropna(how="any")
    returns_df = returns_df.reindex(columns=DEFAULT_TICKERS)
    corr_matrix = returns_df.corr(method="pearson")

    if corr_matrix.shape != (5, 5):
        raise ValueError(f"Expected a 5x5 correlation matrix, got {corr_matrix.shape}.")

    corr_output_path = args.output_dir / "correlation_matrix_5x5.csv"
    corr_matrix.to_csv(corr_output_path, index=True)
    print(f"Saved correlation matrix: {corr_output_path}")


if __name__ == "__main__":
    main()
