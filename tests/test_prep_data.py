
import pandas as pd
import pytest
from src.download_data import get_data, TICKERS
from src.prep_data import prep_data


def test_get_data_and_prep_data():
    # Use a short date range to test the pipeline.
    raw = get_data(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-06-01",
        interval="1d",
    )

    assert not raw.empty, "Downloaded data is empty"

    # Extract AAPL if yfinance returned multi-level columns.
    if isinstance(raw.columns, pd.MultiIndex):
        if "AAPL" in raw.columns.get_level_values(0):
            raw = raw["AAPL"]
        else:
            raw = raw.xs("AAPL", axis=1, level=1)

    assert isinstance(raw.index, pd.DatetimeIndex)
    assert "Close" in raw.columns

    prepared = prep_data(raw.copy())

    assert not prepared.empty
    assert "daily_return" in prepared.columns
    assert "MA_20" in prepared.columns
    assert "MA_50" in prepared.columns
    assert "volatility" in prepared.columns

    # Moving averages and volatility need sufficient history.
    assert prepared["MA_50"].notna().all()

    print("\nPrepared data preview:")
    print(prepared.tail())


def test_prep_data_rejects_non_dataframe():
    with pytest.raises(ValueError):
        prep_data([1, 2, 3])


def test_prep_data_rejects_non_datetime_index():
    df = pd.DataFrame({"Close": [100, 101, 102]})

    with pytest.raises(ValueError):
        prep_data(df)
