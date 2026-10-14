from src import logger



from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Stock Prediction API",
    description="API for ML-based stock price predictions",
    version="1.0.0",
)

MODEL_PATH = "models/stock_model.joblib"
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10,
        examples=["AAPL"],
    )


class PredictionResponse(BaseModel):
    ticker: str
    last_close: float
    predicted_close: float
    predicted_change_pct: float
    forecast_horizon: str
    generated_at: datetime


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep these features identical to the training pipeline."""
    close = data["Close"].dropna()

    if len(close) < 60:
        raise ValueError("At least 60 valid closing prices are required.")

    features = pd.DataFrame(index=close.index)
    features["Close"] = close
    features["Return_1d"] = close.pct_change()
    features["MA_5"] = close.rolling(5).mean()
    features["MA_10"] = close.rolling(10).mean()
    features["MA_20"] = close.rolling(20).mean()
    features["MA_50"] = close.rolling(50).mean()
    features["Volatility_20"] = close.pct_change().rolling(20).std()

    return features.replace([np.inf, -np.inf], np.nan).dropna()


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    ticker = request.ticker.strip().upper()

    try:
        history = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if history.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for ticker {ticker}.",
            )

        # Handle the MultiIndex columns some yfinance versions return.
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        features = create_features(history)
        latest_features = features.iloc[[-1]]

        last_close = float(history["Close"].dropna().iloc[-1])
        predicted_close = float(model.predict(latest_features)[0])

        if not np.isfinite(predicted_close) or predicted_close <= 0:
            raise ValueError("Model returned an invalid predicted price.")

        change_pct = ((predicted_close - last_close) / last_close) * 100

        return PredictionResponse(
            ticker=ticker,
            last_close=round(last_close, 4),
            predicted_close=round(predicted_close, 4),
            predicted_change_pct=round(change_pct, 2),
            forecast_horizon="Next trading day (model assumption)",
            generated_at=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
