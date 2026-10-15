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

OUTPUT_FILENAME = "prepared_stock_data.csv"


def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean one ticker's OHLCV data and create predictive features."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    data = data.copy()

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
    data = data.dropna(subset=list(required))

    if data.empty:
        raise ValueError("No valid OHLCV rows remain after cleaning.")

    # Feature engineering.
    data["MA_Ratio_5"] = data["Close"] / data["Close"].rolling(5).mean() - 1
    data["MA_Ratio_20"] = data["Close"] / data["Close"].rolling(20).mean() - 1
    data["MA_Ratio_50"] = data["Close"] / data["Close"].rolling(50).mean() - 1
    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Volatility_20"] = data["Return_1d"].rolling(20).std()
    data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]
    data["Volume_Change"] = data["Volume"].pct_change()

    # Predict the next trading day's closing price.
    data["Target"] = data["Close"].shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan)

    # Rolling indicators need historical observations.
    data = data.dropna(subset=FEATURE_COLUMNS + ["Target"])

    return data


def resolve_input_file(input_path: Path, ticker: str) -> Path:
    """Azure ML passes inputs as either a file or a mounted folder."""
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        csvs = sorted(input_path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in input folder: {input_path}")
        # Prefer a file named after the ticker, else the first CSV.
        for csv in csvs:
            if ticker.lower() in csv.stem.lower():
                return csv
        return csvs[0]

    raise FileNotFoundError(f"Input not found: {input_path}")


def load_ticker_data(csv_file: Path, ticker: str) -> pd.DataFrame:
    """Load a yfinance multi-ticker CSV or a conventional single-header CSV."""
    ticker_data = None

    try:
        multi = pd.read_csv(csv_file, header=[0, 1], index_col=0)
    except (ValueError, pd.errors.ParserError):
        multi = None

    if multi is not None and isinstance(multi.columns, pd.MultiIndex):
        top = {str(v).upper() for v in multi.columns.get_level_values(0)}
        bottom = {str(v).upper() for v in multi.columns.get_level_values(1)}

        # Match case-insensitively on the level that holds tickers.
        if ticker in top:
            level = 0
        elif ticker in bottom:
            level = 1
        else:
            level = None

        if level is not None:
            upper = multi.columns.get_level_values(level).astype(str).str.upper()
            ticker_data = multi.loc[:, upper == ticker]
            ticker_data.columns = ticker_data.columns.droplevel(level)

    if ticker_data is None:
        # Fall back to a conventional single-header CSV.
        ticker_data = pd.read_csv(csv_file, index_col=0)

    ticker_data.index = pd.to_datetime(ticker_data.index, errors="coerce")
    ticker_data = ticker_data.loc[ticker_data.index.notna()].sort_index()
    return ticker_data


def resolve_output_file(output_path: Path) -> Path:
    """Accept either a .csv file path or a folder (Azure ML output)."""
    if output_path.suffix.lower() == ".csv":
        csv_path = output_path
    else:
        csv_path = output_path / OUTPUT_FILENAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stock data and engineer features."
    )
    parser.add_argument("--input", default="stock_data.csv",
                        help="Input CSV file or folder (Azure ML uri_folder).")
    parser.add_argument("--ticker", default="AAPL",
                        help="Ticker symbol in a multi-ticker yfinance CSV.")
    parser.add_argument("--output", default="data/prepared_stock_data.csv",
                        help="Output CSV path or Azure ML output directory.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    csv_file = resolve_input_file(Path(args.input), ticker)
    ticker_data = load_ticker_data(csv_file, ticker)

    prepared = prep_data(ticker_data)

    csv_path = resolve_output_file(Path(args.output))
    prepared.to_csv(csv_path, index_label="Date")

    print(f"Input: {csv_file}")
    print(f"Ticker: {ticker}")
    print(f"Prepared dataset shape: {prepared.shape}")
    print(f"Saved to: {csv_path}")
    print(f"Features: {FEATURE_COLUMNS}")
    print("Target: next trading day's Close")


if __name__ == "__main__":
    main()