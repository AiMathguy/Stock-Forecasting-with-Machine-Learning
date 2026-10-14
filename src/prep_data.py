
"""Clean stock OHLCV data and engineer features for model training."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Return_1d",
    "Return_5d",
    "MA_Ratio_5",
    "MA_Ratio_20",
    "MA_Ratio_50",
    "Volatility_20",
    "High_Low_Range",
    "Volume_Change",
]


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean one ticker's OHLCV data and create predictive features."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    data = data.copy()

    # Handle yfinance multi-level columns if present.
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError(
            "Input must contain one ticker only. Select a ticker before prep_data()."
        )

    # Normalise column names.
    data.columns = [str(col).strip().title() for col in data.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    # Ensure chronological ordering and valid numeric values.
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("Data index must contain valid dates.") from exc

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if data.empty:
        raise ValueError("No valid OHLCV rows remain after cleaning.")

    # Feature engineering.
    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)

    data["MA_Ratio_5"] = (
        data["Close"] / data["Close"].rolling(5).mean()
    )
    data["MA_Ratio_20"] = (
        data["Close"] / data["Close"].rolling(20).mean()
    )
    data["MA_Ratio_50"] = (
        data["Close"] / data["Close"].rolling(50).mean()
    )

    data["Volatility_20"] = (
        data["Return_1d"].rolling(20).std()
    )

    data["High_Low_Range"] = (
        (data["High"] - data["Low"]) / data["Close"]
    )

    data["Volume_Change"] = data["Volume"].pct_change()

    # Predict the next trading day's closing price.
    data["Target"] = data["Close"].shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan)

    # Rolling indicators need historical observations.
    data = data.dropna(subset=FEATURE_COLUMNS + ["Target"])

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stock data and engineer features."
    )
    parser.add_argument(
        "--input",
        default="src/stock_data.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol in a multi-ticker yfinance CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/prepared_stock_data.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    input_path = Path(args.input)

    if not input_path.is_absolute():
        # First check relative to the current working directory.
        if not input_path.exists():
            # Then check relative to this script's directory.
            input_path = Path(__file__).resolve().parent / input_path.name

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")


    # Multi-ticker CSV saved by yfinance has a two-row header.
    raw = pd.read_csv(
        input_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(
            "Expected a multi-ticker CSV with two header rows."
        )

    ticker = args.ticker.upper()
    top_level = {str(value).upper() for value in raw.columns.get_level_values(0)}
    bottom_level = {str(value).upper() for value in raw.columns.get_level_values(1)}

    if ticker in top_level:
        ticker_data = raw.xs(ticker, axis=1, level=0)
    elif ticker in bottom_level:
        ticker_data = raw.xs(ticker, axis=1, level=1)
    else:
        raise ValueError(f"Ticker {ticker!r} was not found in {input_path}.")

    prepared = prep_data(ticker_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index_label="Date")

    print(f"Ticker: {ticker}")
    print(f"Prepared dataset shape: {prepared.shape}")
    print(f"Saved to: {output_path}")
    print(f"Features: {FEATURE_COLUMNS}")
    print("Target: next trading day's Close")


if __name__ == "__main__":
    main()
