from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

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
    parser.add_argument("--tickers",     nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--windows",     nargs="+", type=int, default=[30])
    parser.add_argument("--min-periods", type=int,  default=30)
    parser.add_argument(
        "--plot-rolling-vol",
        action="store_true",
        help="Generate a multi-panel rolling volatility figure for the selected tickers.",
    )
    parser.add_argument(
        "--subperiod-corr",
        action="store_true",
        help="Compute sub-period correlation matrices and a side-by-side heatmap figure.",
    )
    parser.add_argument(
        "--macro-corr",
        action="store_true",
        help="Compute correlation table between FRED macro series and rolling volatility.",
    )
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


def plot_rolling_volatility_panels(
    rolling_vols: dict[str, dict[int, pd.Series]],
    output_path: Path,
) -> None:
    tickers = list(rolling_vols.keys())
    windows = sorted({w for vols in rolling_vols.values() for w in vols})
    if not tickers or not windows:
        return

    n_rows = len(tickers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    color_map = {30: "#1f77b4", 60: "#ff7f0e", 90: "#2ca02c"}
    for ax, ticker in zip(axes, tickers):
        for window in windows:
            series = rolling_vols[ticker].get(window)
            if series is None:
                continue
            ax.plot(series.index, series.values, label=f"{window}d", color=color_map.get(window))

        ax.set_title(ticker)
        ax.grid(True, alpha=0.3)

        # Highlight key market regimes for context.
        ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-31"),
                   color="#d62728", alpha=0.08, label="COVID crash")
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
                   color="#1f77b4", alpha=0.05, label="2021 recovery")
        ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"),
                   color="#9467bd", alpha=0.05, label="2022–2023 rate hikes")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        leg_y = 0.985 if n_rows > 10 else 0.95
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, leg_y), ncol=6, frameon=False)

    title_y = 0.995 if n_rows > 10 else 0.98
    fig.suptitle("Rolling Volatility (30/60/90-day) Across Key Market Regimes", y=title_y)
    
    rect_top = 0.965 if n_rows > 10 else 0.92
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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


def compute_subperiod_correlations(
    all_lr: dict[str, pd.Series],
    tickers: list[str],
    output_dir: Path,
) -> list[tuple[str, pd.DataFrame]]:
    if len(all_lr) < 2:
        return []

    lr_df = pd.DataFrame(all_lr).dropna()
    n = len(tickers)

    subperiods = [
        ("2019", "2019-01-01", "2019-12-31", f"correlation_matrix_{n}x{n}_2019.csv"),
        ("Mar 2020", "2020-03-01", "2020-03-31", f"correlation_matrix_{n}x{n}_mar2020.csv"),
        ("2022–2023", "2022-01-01", "2023-12-31", f"correlation_matrix_{n}x{n}_2022_2023.csv"),
    ]

    results: list[tuple[str, pd.DataFrame]] = []
    for label, start, end, filename in subperiods:
        window_df = lr_df.loc[start:end].dropna()
        if window_df.empty:
            print(f"  - Skipping {label}: no overlapping data in range {start} to {end}")
            continue
        corr = window_df.corr().reindex(index=tickers, columns=tickers)
        corr.to_csv(output_dir / filename)
        results.append((label, corr))
        print(f"  Saved {label} correlation matrix -> {filename}")

    return results


def plot_subperiod_correlation_heatmaps(
    corr_matrices: list[tuple[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    if not corr_matrices:
        return

    fig, axes = plt.subplots(1, len(corr_matrices), figsize=(6.5 * len(corr_matrices), 6))
    if len(corr_matrices) == 1:
        axes = [axes]

    for ax, (label, corr) in zip(axes, corr_matrices):
        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            square=True,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Correlation")
    fig.suptitle("Cross-Asset Correlation Shifts Across Market Regimes", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_macro_correlations(
    all_lr: dict[str, pd.Series],
    data_dir: Path,
    output_dir: Path,
    window: int = 30,
    min_periods: int = 30,
) -> pd.DataFrame:
    print(f"\nComputing correlations between FRED macro series and {window}-day rolling volatility...")
    
    # Load and merge FRED macro series
    macro_series_names = ["CPIAUCSL", "DGS10", "UNRATE"]
    macro_dfs = []
    for name in macro_series_names:
        csv_path = data_dir / f"{name}_fred.csv"
        if not csv_path.exists():
            print(f"  - Missing FRED file: {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").rename(columns={"value": name})
        macro_dfs.append(df)
        
    if not macro_dfs:
        print("  - No macro data found. Skipping macro correlation analysis.")
        return pd.DataFrame()
        
    macro_df = pd.concat(macro_dfs, axis=1, join="inner")
    
    rows = []
    for ticker, lr in all_lr.items():
        # Compute rolling volatility
        vol = compute_rolling_volatility(lr, window, min_periods).dropna()
        # Convert vol index to datetime to align with macro_df
        vol_df = pd.DataFrame({"vol": vol})
        vol_df.index = pd.to_datetime(vol_df.index)
        
        # Merge with macro data
        merged = vol_df.join(macro_df, how="inner")
        if merged.empty:
            continue
            
        corr_row = {"Ticker": ticker}
        for name in macro_series_names:
            if name in merged.columns:
                corr_val = merged["vol"].corr(merged[name])
                corr_row[name] = round(corr_val, 4)
        rows.append(corr_row)
        
    if not rows:
        print("  - No overlapping dates between rolling volatility and macro series.")
        return pd.DataFrame()
        
    corr_df = pd.DataFrame(rows)
    # Reorder columns to ensure consistency
    cols = ["Ticker"] + [name for name in macro_series_names if name in corr_df.columns]
    corr_df = corr_df[cols]
    
    output_path = output_dir / "macro_volatility_correlation.csv"
    corr_df.to_csv(output_path, index=False)
    print(f"  Saved macro correlation table -> {output_path.name}")
    return corr_df


def main() -> None:
    args       = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing metrics for {len(args.tickers)} equities...")
    all_lr: dict[str, pd.Series] = {}
    rolling_vols: dict[str, dict[int, pd.Series]] = {}
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

            all_lr[ticker] = lr

            # Save log returns
            lr_out = pd.DataFrame({"Date": lr.index, "Log_Return": lr.values})
            lr_out.to_csv(output_dir / f"{safe_ticker}_log_returns.csv", index=False)

            # Save rolling volatility for each requested window
            rolling_vols[ticker] = {}
            for window in args.windows:
                vol = compute_rolling_volatility(lr, window, args.min_periods)
                rolling_vols[ticker][window] = vol
                vol_out = pd.DataFrame(
                    {
                        "Date": vol.index,
                        f"Rolling_Vol_{window}d": vol.values,
                    }
                )
                vol_out.to_csv(
                    output_dir / f"{safe_ticker}_rolling_volatility_{window}d.csv",
                    index=False,
                )

            summary_rows.append(compute_summary_stats(lr, ticker))
            print(f"  + {ticker}")
        except Exception as exc:
            failed.append(ticker)
            print(f"  - {ticker}: {exc}")

    # Cross-asset correlation matrix (NxN, aligned on common dates)
    if len(all_lr) > 1:
        lr_df = pd.DataFrame(all_lr).dropna()
        corr_matrix = lr_df.corr()
        n = len(all_lr)
        corr_matrix.to_csv(output_dir / f"correlation_matrix_{n}x{n}.csv")
        print(f"\n  Saved {n}x{n} correlation matrix -> correlation_matrix_{n}x{n}.csv")

    # Summary statistics table
    if summary_rows:
        stats_df = pd.DataFrame(summary_rows)
        stats_df.to_csv(output_dir / "summary_stats_all_equities.csv", index=False)
        print(f"  Saved summary stats -> summary_stats_all_equities.csv")

    if args.plot_rolling_vol:
        output_path = output_dir / f"rolling_volatility_panels_{len(all_lr)}_equities.png"
        plot_rolling_volatility_panels(rolling_vols, output_path)
        print(f"  Saved rolling volatility panels -> {output_path.name}")

    if args.subperiod_corr:
        corr_matrices = compute_subperiod_correlations(all_lr, args.tickers, output_dir)
        output_path = output_dir / f"correlation_heatmaps_subperiods_{len(all_lr)}_equities.png"
        plot_subperiod_correlation_heatmaps(corr_matrices, output_path)
        print(f"  Saved sub-period heatmaps -> {output_path.name}")

    if args.macro_corr:
        window = args.windows[0] if args.windows else 30
        compute_macro_correlations(all_lr, args.data_dir, output_dir, window, args.min_periods)

    if failed:
        print(f"\nFailed tickers: {failed}")
    else:
        print(f"\nAll {len(args.tickers)} equities processed successfully.")


if __name__ == "__main__":
    main()
