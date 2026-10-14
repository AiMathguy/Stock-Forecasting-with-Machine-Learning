import os
import zipfile
import ast
from pathlib import Path

# Resolve the stock-trading project root on Windows.
base = Path(__file__).resolve().parent.parent
src = base / "src"
src.mkdir(parents=True, exist_ok=True)

downloader = r'''"""Download historical OHLCV data for multiple tickers into a shared CSV."""

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
'''

(src / "download_data.py").write_text(downloader, encoding="utf-8")

train_path = src / "train.py"
train = train_path.read_text(encoding="utf-8")

start = train.index("def download_data(")
end = train.index("\n\ndef make_dataset(", start)

new_func = r'''def download_data(ticker: str, period: str, data_csv: str | None = None) -> pd.DataFrame:
    """Load ticker data from a shared CSV or download it directly."""
    if data_csv:
        if not os.path.exists(data_csv):
            raise FileNotFoundError(f"CSV not found: {data_csv}")

        data = pd.read_csv(
            data_csv, header=[0, 1], index_col=0, parse_dates=True
        )

        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Expected a multi-ticker CSV with a two-row header.")

        top = {str(v).upper() for v in data.columns.get_level_values(0)}
        bottom = {str(v).upper() for v in data.columns.get_level_values(1)}

        if ticker.upper() in top:
            data = data.xs(ticker.upper(), axis=1, level=0)
        elif ticker.upper() in bottom:
            data = data.xs(ticker.upper(), axis=1, level=1)
        else:
            raise ValueError(f"Ticker {ticker!r} not found in {data_csv}.")
    else:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise ValueError(f"No data returned for {ticker!r}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).strip().title() for c in data.columns]

    required = {"Close", "High", "Low", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = data.sort_index()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["Close", "High", "Low"])
'''

train = train[:start] + new_func + train[end:]

train = train.replace(
    'def train(ticker: str, period: str, output_dir: str) -> dict[str, Any]:\n'
    '    os.makedirs(output_dir, exist_ok=True)\n'
    '    raw = download_data(ticker, period)',
    'def train(ticker: str, period: str, output_dir: str, data_csv: str | None = None) -> dict[str, Any]:\n'
    '    os.makedirs(output_dir, exist_ok=True)\n'
    '    raw = download_data(ticker, period, data_csv=data_csv)'
)

train = train.replace(
    'parser.add_argument("--output-dir", default="models", help="Directory for model and report")\n'
    '    return parser.parse_args()',
    'parser.add_argument("--output-dir", default="models", help="Directory for model and report")\n'
    '    parser.add_argument("--data-csv", default=None, help="Path to shared stock CSV")\n'
    '    return parser.parse_args()'
)

train = train.replace(
    'train(args.ticker, args.period, args.output_dir)',
    'train(args.ticker, args.period, args.output_dir, data_csv=args.data_csv)'
)

train_path.write_text(train, encoding="utf-8")

requirements = """yfinance
pandas
numpy
scipy
scikit-learn
joblib
fastapi
uvicorn[standard]
"""

(base / "requirements.txt").write_text(requirements, encoding="utf-8")

readme = """# Stock Prediction Project

## Install dependencies
```bash
pip install -r requirements.txt
```

## Download stock data
Run from the project root:
```bash
python src/download_data.py --start 2020-01-01 --end 2024-01-01 --output src/stock_data.csv
```

## Train using the shared CSV
```bash
python src/train.py --ticker AAPL --data-csv src/stock_data.csv
```

## Train using a direct download
```bash
python src/train.py --ticker AAPL --period 5y
```
"""

(base / "README.md").write_text(readme, encoding="utf-8")

for path in [train_path, src / "download_data.py"]:
    ast.parse(path.read_text(encoding="utf-8"))

zip_path = base / "stock_prediction_connected.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__"}]
        for name in files:
            path = Path(root) / name
            if path != zip_path:
                z.write(path, path.relative_to(base))

print(f"Project updated: {base}")
print(f"ZIP created: {zip_path}")
print(f"Train supports --data-csv: {'--data-csv' in train}")
