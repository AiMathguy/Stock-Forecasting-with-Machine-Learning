# Stock Prediction Project

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
