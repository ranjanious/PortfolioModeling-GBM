from __future__ import annotations

from pathlib import Path
import argparse
import os
import time

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


DEFAULT_SERIES = ["UNRATE", "CPIAUCSL", "DGS10"]


def clean_series(series: pd.Series, target_freq: str) -> pd.DataFrame:
    if series.empty:
        raise ValueError("Received empty FRED series.")

    cleaned = series.dropna().copy()
    if cleaned.empty:
        raise ValueError("FRED series only contains missing values.")

    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()]
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    full_index = pd.date_range(cleaned.index.min(), cleaned.index.max(), freq=target_freq)
    cleaned = cleaned.reindex(full_index)
    cleaned = cleaned.ffill().bfill()

    df = cleaned.rename("value").to_frame().reset_index(names="Date")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def fetch_series(fred: Fred, series_id: str, retries: int) -> pd.Series:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            series = fred.get_series(series_id)
            if series.empty:
                raise ValueError(f"No data returned for series '{series_id}'.")
            return series
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"Retrying {series_id} (attempt {attempt + 1}/{retries})...")
                time.sleep(2 * attempt)

    raise RuntimeError(f"FRED failed for series '{series_id}' after {retries} attempts: {last_error}")


def save_csv(df: pd.DataFrame, series_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{series_id.upper()}_fred.csv"
    df.to_csv(output_path, index=False)
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FRED series and save cleaned no-gap CSVs to /data."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=DEFAULT_SERIES,
        help="Space-separated FRED series IDs. Default: UNRATE CPIAUCSL DGS10",
    )
    parser.add_argument(
        "--freq",
        default="B",
        help="Target index frequency for gap-filling (pandas offset alias). Default: B (business day).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per series on API/network error. Default: 3",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found. Add it to a .env file in the project root.")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"

    fred = Fred(api_key=api_key)

    print("Starting FRED download...")
    print(f"Series: {', '.join(s.upper() for s in args.series)}")
    print(f"Gap-fill frequency: {args.freq}")

    failed_series: list[str] = []

    for series_id in args.series:
        sid = series_id.upper()
        try:
            series = fetch_series(fred, sid, retries=args.retries)
            cleaned_df = clean_series(series, target_freq=args.freq)
            saved_path = save_csv(cleaned_df, sid, output_dir)
            print(f"Saved {sid}: {saved_path}")
        except Exception as exc:
            failed_series.append(sid)
            print(f"Failed {sid}: {exc}")

    if failed_series:
        failed_csv = ", ".join(failed_series)
        raise RuntimeError(f"Completed with failures. Could not fetch: {failed_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
