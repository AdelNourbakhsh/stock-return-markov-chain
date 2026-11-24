"""
Fetch historical daily AAPL prices and discretize daily returns into a 7-state
Markov chain dataset.

Usage:
    export FMP_API_KEY="your_api_key_here"          # Linux / macOS
    set FMP_API_KEY=your_api_key_here               # Windows CMD
    $env:FMP_API_KEY="your_api_key_here"            # Windows PowerShell

    python fetch_data.py --symbol AAPL --start 2019-07-23 --end 2025-07-18 --output data/AAPL_states.csv
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd


def classify_return(r: float) -> str:
    """Map daily percentage return to one of seven discrete return regimes."""
    if r < -2.0:
        return "R1"
    if -2.0 <= r < -1.0:
        return "R2"
    if -1.0 <= r < 0.0:
        return "R3"
    if r == 0.0:
        return "R4"
    if 0.0 < r <= 1.0:
        return "R5"
    if 1.0 < r <= 2.0:
        return "R6"
    return "R7"


def fetch_historical_prices(symbol: str, api_key: str) -> pd.DataFrame:
    """Download daily close prices from Financial Modeling Prep API."""
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        f"?serietype=line&apikey={api_key}"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "historical" not in payload:
        raise RuntimeError(f"Unexpected API response for symbol '{symbol}'.")

    df = pd.DataFrame(payload["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["close"]].rename(columns={"close": "Adj Close"})


def build_state_dataset(
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    api_key: str,
) -> pd.DataFrame:
    df = fetch_historical_prices(symbol, api_key)

    # Compute daily percentage returns before slicing to preserve the first date's return
    df["Return"] = df["Adj Close"].pct_change() * 100.0
    df = df.dropna()

    df = df.loc[start_date:end_date].copy()
    df["State"] = df["Return"].apply(classify_return)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical stock prices and discretize daily returns into Markov states."
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2019-07-23", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-07-18", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="data/AAPL_states.csv",
        help="Output CSV path (default: data/AAPL_states.csv)",
    )
    args = parser.parse_args()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError(
            "Environment variable FMP_API_KEY is not set. "
            "Set your Financial Modeling Prep API key or run markov_analysis.py "
            "directly on the included data/AAPL_states.csv file."
        )

    output_path = Path(args.output)
    df = build_state_dataset(args.symbol, args.start, args.end, output_path, api_key)
    print(f"Saved {len(df):,} trading days ({df.index.min().date()} to {df.index.max().date()}) to {output_path}")


if __name__ == "__main__":
    main()
