"""Download historical OHLCV data for multiple tickers into a shared CSV."""

import argparse
import os
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
    "JPM", "GS", "V", "MA",
    "TSLA", "WMT", "COST", "KO", "MCD",
    "LLY", "JNJ", "UNH",
    "SPY", "QQQ", "IWM", "DIA",
    "GLD", "SLV", "USO", "^VIX",
]


def get_data(tickers, start_date, end_date, interval="1d"):
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )
    if data.empty:
        raise ValueError("No data downloaded. Check tickers, dates, or network access.")
    return data.sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output", default="src/stock_data.csv")
    args = parser.parse_args()

    data = get_data(TICKERS, args.start, args.end, args.interval)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    data.to_csv(args.output)
    print(f"Saved {data.shape[0]} rows x {data.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
