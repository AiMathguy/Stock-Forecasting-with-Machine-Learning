"""Train the stock price prediction model."""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import os, textwrap
base = "/mnt/data/stock_prediction_project"
os.makedirs(os.path.join(base, "src"), exist_ok=True)
content = r"""
Stock prediction training pipeline with data-characteristics voting.

Usage:
    python src/train.py --ticker AAPL
    python src/train.py --ticker MSFT --period 5y

Target:
    Next trading day's adjusted closing price.

Notes:
- This is an educational modelling pipeline, not financial advice.
- Features and target are shifted so each row only uses information available
  at that row's close. Evaluation is chronological.
- Data-characteristic voting is a heuristic. Time-series validation decides
  which of the selected models performs best.
"""
"""Stock price prediction training pipeline."""


import argparse
import numpy as np
import pandas as pd

import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import kurtosis, skew
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURE_COLUMNS = [
    "Close",
    "Return_1d",
    "Return_5d",
    "MA_Ratio_5",
    "MA_Ratio_20",
    "MA_Ratio_50",
    "Volatility_20",
    "High_Low_Range",
    "Volume_Change",
]


@dataclass
class DataProfile:
    rows: int
    feature_count: int
    return_skewness: float
    return_excess_kurtosis: float
    outlier_return_fraction: float
    return_volatility: float
    nonlinear_signal: bool
    small_sample: bool
    high_volatility: bool
    heavy_tails: bool
    skewed_returns: bool


def download_data(ticker: str, period: str, data_csv: str | None = None) -> pd.DataFrame:
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


def make_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create causal features and next-session close target."""
    close = raw["Close"].astype(float)
    volume = raw["Volume"].astype(float).replace(0, np.nan)

    features = pd.DataFrame(index=raw.index)
    features["Close"] = close
    features["Return_1d"] = close.pct_change(1)
    features["Return_5d"] = close.pct_change(5)
    features["MA_Ratio_5"] = close / close.rolling(5).mean() - 1
    features["MA_Ratio_20"] = close / close.rolling(20).mean() - 1
    features["MA_Ratio_50"] = close / close.rolling(50).mean() - 1
    features["Volatility_20"] = close.pct_change().rolling(20).std()
    features["High_Low_Range"] = (raw["High"] - raw["Low"]) / close
    features["Volume_Change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)

    target = close.shift(-1).rename("Target_Next_Close")
    combined = features.replace([np.inf, -np.inf], np.nan).join(target).dropna()
    if len(combined) < 100:
        raise ValueError(
            f"Only {len(combined)} usable rows remain. Use a longer period (e.g. --period 5y)."
        )
    return combined[FEATURE_COLUMNS], combined["Target_Next_Close"]


def profile_data(X: pd.DataFrame, y: pd.Series) -> DataProfile:
    returns = X["Return_1d"].dropna()
    if returns.empty:
        raise ValueError("Cannot profile empty returns.")

    ret_skew = float(skew(returns, bias=False, nan_policy="omit"))
    ret_kurt = float(kurtosis(returns, fisher=True, bias=False, nan_policy="omit"))
    median = float(returns.median())
    mad = float(np.median(np.abs(returns - median)))
    # Robust modified z-score; protect against a near-zero MAD.
    if mad > 1e-12:
        outlier_fraction = float((np.abs(returns - median) > 3.5 * 1.4826 * mad).mean())
    else:
        outlier_fraction = 0.0
    volatility = float(returns.std())

    # Heuristic signal check: compare a linear relationship to a nonlinear
    # tree model on a chronological holdout. This is diagnostic, not a final score.
    split = max(int(len(X) * 0.8), 1)
    nonlinear_signal = False
    if len(X) - split >= 20 and split >= 50:
        try:
            from sklearn.metrics import mean_absolute_error
            lin = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
            tree = HistGradientBoostingRegressor(max_iter=100, random_state=42)
            lin.fit(X.iloc[:split], y.iloc[:split])
            tree.fit(X.iloc[:split], y.iloc[:split])
            lin_mae = mean_absolute_error(y.iloc[split:], lin.predict(X.iloc[split:]))
            tree_mae = mean_absolute_error(y.iloc[split:], tree.predict(X.iloc[split:]))
            nonlinear_signal = bool(tree_mae < lin_mae * 0.97)
        except Exception:
            nonlinear_signal = False

    return DataProfile(
        rows=int(len(X)),
        feature_count=int(X.shape[1]),
        return_skewness=ret_skew if np.isfinite(ret_skew) else 0.0,
        return_excess_kurtosis=ret_kurt if np.isfinite(ret_kurt) else 0.0,
        outlier_return_fraction=outlier_fraction,
        return_volatility=volatility if np.isfinite(volatility) else 0.0,
        nonlinear_signal=nonlinear_signal,
        small_sample=len(X) < 500,
        high_volatility=volatility > 0.025,
        heavy_tails=ret_kurt > 3.0,
        skewed_returns=abs(ret_skew) > 1.0,
    )


def vote_for_models(profile: DataProfile) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Assign heuristic votes to candidate regressors based on return distribution,
    outliers, volatility, sample size and a nonlinear-signal diagnostic.
    """
    votes = {
        "Ridge": 0,
        "SVR_RBF": 0,
        "RandomForest": 0,
        "ExtraTrees": 0,
        "HistGradientBoosting": 0,
    }
    reasons: dict[str, list[str]] = {name: [] for name in votes}

    def award(model: str, points: int, reason: str) -> None:
        votes[model] += points
        reasons[model].append(reason)

    # Baseline/regularised linear model gets a vote for noisy or small datasets.
    award("Ridge", 1, "regularised linear baseline")
    if profile.small_sample:
        award("Ridge", 2, "smaller sample favours a lower-complexity baseline")
        award("SVR_RBF", 1, "kernel model can work on modest-sized datasets")
    else:
        award("RandomForest", 1, "ample rows support a nonlinear ensemble")
        award("ExtraTrees", 1, "ample rows support a nonlinear ensemble")
        award("HistGradientBoosting", 1, "ample rows support boosting")

    if profile.nonlinear_signal:
        award("RandomForest", 2, "nonlinear diagnostic outperformed linear baseline")
        award("ExtraTrees", 2, "nonlinear diagnostic outperformed linear baseline")
        award("HistGradientBoosting", 2, "nonlinear diagnostic outperformed linear baseline")
        award("SVR_RBF", 1, "nonlinear signal supports testing an RBF kernel")
    else:
        award("Ridge", 1, "no clear nonlinear advantage in diagnostic split")

    if profile.heavy_tails or profile.outlier_return_fraction > 0.02:
        award("ExtraTrees", 2, "robust tree ensembles can handle irregular/outlier patterns")
        award("RandomForest", 1, "tree ensemble candidate for irregular observations")
        award("SVR_RBF", 1, "scaled kernel model retained as a comparison")
    if profile.skewed_returns:
        award("ExtraTrees", 1, "skewed returns motivate a flexible nonlinear candidate")
        award("HistGradientBoosting", 1, "skewed returns motivate a boosting candidate")
    if profile.high_volatility:
        award("RandomForest", 1, "high volatility supports testing nonlinear interactions")
        award("HistGradientBoosting", 1, "high volatility supports testing boosting")
        award("SVR_RBF", 1, "scaled RBF model retained for volatile data")

    # Deterministic tie-break by name; choose exactly three candidates.
    ranked = sorted(votes, key=lambda name: (-votes[name], name))
    selected_votes = {name: votes[name] for name in ranked[:3]}
    selected_reasons = {name: reasons[name] for name in ranked[:3]}
    return selected_votes, selected_reasons


def candidate_models() -> dict[str, Any]:
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "SVR_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=42
            )),
        ]),
    }


def evaluate_chronologically(
    model: Any, X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.2
) -> dict[str, float]:
    split = int(len(X) * (1 - test_fraction))
    if split < 50 or len(X) - split < 20:
        raise ValueError("Not enough rows for a chronological train/test split.")

    fitted = clone(model)
    fitted.fit(X.iloc[:split], y.iloc[:split])
    predictions = fitted.predict(X.iloc[split:])
    mae = float(mean_absolute_error(y.iloc[split:], predictions))
    rmse = float(np.sqrt(mean_squared_error(y.iloc[split:], predictions)))
    # Direction accuracy compares predicted and actual close-to-close change.
    actual_direction = np.sign(y.iloc[split:].to_numpy() - X["Close"].iloc[split:].to_numpy())
    predicted_direction = np.sign(predictions - X["Close"].iloc[split:].to_numpy())
    direction_accuracy = float(np.mean(actual_direction == predicted_direction))
    return {
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "test_rows": int(len(predictions)),
    }


def train(ticker: str, period: str, output_dir: str, data_csv: str | None = None) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    raw = download_data(ticker, period, data_csv=data_csv)
    X, y = make_dataset(raw)
    profile = profile_data(X, y)
    votes, vote_reasons = vote_for_models(profile)

    all_models = candidate_models()
    scores: dict[str, dict[str, float]] = {}
    for model_name in votes:
        scores[model_name] = evaluate_chronologically(all_models[model_name], X, y)

    # Select lowest chronological MAE, then RMSE as a tie-breaker.
    winner_name = min(scores, key=lambda name: (scores[name]["mae"], scores[name]["rmse"]))
    winner = clone(all_models[winner_name])
    winner.fit(X, y)

    model_path = os.path.join(output_dir, "stock_model.joblib")
    joblib.dump(winner, model_path)

    report: dict[str, Any] = {
        "ticker": ticker.upper(),
        "period": period,
        "target": "next_trading_day_close",
        "selected_models_by_vote": votes,
        "vote_reasons": vote_reasons,
        "data_profile": asdict(profile),
        "chronological_holdout_metrics": scores,
        "winner_by_holdout_mae": winner_name,
        "saved_model_path": model_path,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X)),
        "note": (
            "Holdout metrics are for one chronological split. For stronger evidence, "
            "add expanding-window walk-forward validation and compare against a naive "
            "baseline that predicts the next close equals the latest close."
        ),
    }
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("\n=== Data profile ===")
    print(json.dumps(report["data_profile"], indent=2))
    print("\n=== Top 3 models selected by voting ===")
    for name, vote_count in votes.items():
        print(f"{name}: {vote_count} votes")
        for reason in vote_reasons[name]:
            print(f"  - {reason}")
    print("\n=== Chronological holdout metrics ===")
    for name, metrics in sorted(scores.items(), key=lambda item: item[1]["mae"]):
        print(
            f"{name}: MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"direction_accuracy={metrics['direction_accuracy']:.3f}"
        )
    print(f"\nWinner: {winner_name}")
    print(f"Saved model: {model_path}")
    print(f"Training report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vote-select and train stock prediction models.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker, e.g. AAPL or MSFT")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 1y, 2y, 5y, max")
    parser.add_argument("--output-dir", default="models", help="Directory for model and report")
    parser.add_argument("--data-csv", default=None, help="Path to shared stock CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.ticker, args.period, args.output_dir, data_csv=args.data_csv)

path = os.path.join(base, "src", "train.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(path)
