
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Stock Forecasting API",
    description="API for stock price predictions",
    version="1.0.0",
)


class TickerInput(BaseModel):
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        examples=["AAPL"],
    )


@app.get("/")
def home():
    return {"status": "online", "message": "Stock Forecasting API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: TickerInput):
    ticker = data.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=422,
            detail="Ticker cannot be empty",
        )

    # TODO: Load the trained model and generate a prediction.
    return {
        "ticker": ticker,
        "prediction": None,
        "status": "model_not_connected",
    }
