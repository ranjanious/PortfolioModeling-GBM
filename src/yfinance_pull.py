from __future__ import annotations

from pathlib import Path
import argparse
import time

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if cleaned.empty:
        raise ValueError("Received empty price data after download.")

    # Ensure index is datetime, timezone-naive, sorted, and unique.
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()]
    if getattr(cleaned.index, "tz", None) is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    # Keep expected OHLCV columns in a stable order.
    available = [col for col in REQUIRED_COLUMNS if col in cleaned.columns]
    cleaned = cleaned[available]

    # Reindex to business days to make date gaps explicit, then fill.
    full_business_days = pd.date_range(cleaned.index.min(), cleaned.index.max(), freq="B")
    cleaned = cleaned.reindex(full_business_days)
    cleaned.index.name = "Date"

    # Fill missing values in a conservative order.
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.reset_index()
    cleaned["Date"] = cleaned["Date"].dt.strftime("%Y-%m-%d")

    return cleaned


def download_ticker_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            # Primary path: batch-style download.
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=30,
            )

            # Fallback path: direct ticker history.
            if df.empty:
                df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            # yfinance can return a MultiIndex in some cases; flatten if needed.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            return clean_ohlcv(df)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                print(f"Retrying {ticker} (attempt {attempt + 1}/3)...")
                time.sleep(2 * attempt)

    raise RuntimeError(f"yfinance failed for ticker '{ticker}' after 3 attempts: {last_error}")


def save_csv(df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker.upper()}_daily_5y.csv"
    df.to_csv(output_path, index=False)
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 5-year daily OHLCV data for selected tickers and save cleaned CSVs to /data."
    )
    parser.add_argument(
        "--tickers",
        nargs='+',
        default=DEFAULT_TICKERS,
        help="Space-separated ticker symbols. Default: AAPL MSFT GOOGL AMZN TSLA",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period window. Default: 5y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval. Default: 1d",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args([])

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"

    print("Starting yfinance download...")
    print(f"Tickers: {', '.join(t.upper() for t in args.tickers)}")

    failed_tickers: list[str] = []

    for ticker in args.tickers:
        ticker_upper = ticker.upper()
        try:
            cleaned_df = download_ticker_data(ticker_upper, args.period, args.interval)
            saved_path = save_csv(cleaned_df, ticker_upper, output_dir)
            print(f"Saved {ticker_upper}: {saved_path}")
        except Exception as exc:
            failed_tickers.append(ticker_upper)
            print(f"Failed {ticker_upper}: {exc}")

    if failed_tickers:
        failed_csv = ", ".join(failed_tickers)
        raise RuntimeError(f"Completed with failures. Could not fetch: {failed_csv}")

    print("Done.")


if __name__ == "__main__":
    main()