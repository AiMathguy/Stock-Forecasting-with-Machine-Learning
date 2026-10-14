
from __future__ import annotations
from pathlib import Path



content = r"""
Hyperparameter tuning and chronological evaluation for the stock prediction pipeline.

Run from the project root:
    python src/eval.py --ticker AAPL --period 5y
    python src/eval.py --ticker AAPL --data-csv data/stock_data.csv

The latest 20% of rows is reserved as a chronological holdout. GridSearchCV uses
TimeSeriesSplit on the earlier development segment only. The winner is selected
by cross-validation MAE, not by holdout performance.
"""

import mlflow
import optuna
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from train import download_data, make_dataset, profile_data, vote_for_models

# Match the eight-feature schema requested for the prediction model.
MODEL_FEATURES = [
    "Return_1d",
    "Return_5d",
    "MA_Ratio_5",
    "MA_Ratio_20",
    "MA_Ratio_50",
    "Volatility_20",
    "High_Low_Range",
    "Volume_Change",
]


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# authenticate
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception:
    credential = InteractiveBrowserCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="subscription_id",
    resource_group_name="resource_group",
    workspace_name="workspace_name",
)

def load_csv_for_ticker(path, ticker):
    import pandas as pd

    # This CSV has two MultiIndex header rows plus an extra Date row.
    df = pd.read_csv(
        path,
        header=[0, 1],
        index_col=0,
        skiprows=[2],
    )

    # Select the requested ticker from the MultiIndex columns.
    available_tickers = df.columns.get_level_values(0).unique().tolist()

    if ticker not in available_tickers:
        raise ValueError(
            f"Ticker {ticker!r} not found in CSV. "
            f"Available tickers: {available_tickers}"
        )

    raw = df[ticker].copy()
    raw.columns = [str(col).strip().title() for col in raw.columns]

    # Convert the index to dates.
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw.index.name = "Date"
    raw = raw.loc[raw.index.notna()]

    # Retain the fields needed for feature engineering.
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in raw.columns]

    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")

    raw = raw[required].apply(pd.to_numeric, errors="coerce")
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])

    if raw.empty:
        raise ValueError(f"No valid price rows found for {ticker!r}.")

    print(f"Loaded {len(raw)} rows for {ticker} from {path}")
    print(f"Date range: {raw.index.min().date()} to {raw.index.max().date()}")

    return raw


def tune_with_optuna(model_name, X_train, y_train, n_trials=30):
    cv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        if model_name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 400, step=100
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 10, 20, 30]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 5
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", [1.0, "sqrt"]
                ),
                "random_state": 42,
                "n_jobs": 1,
            }
            model = ExtraTreesRegressor(**params)

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 400, step=100
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 10, 20, 30]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 5
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", [1.0, "sqrt"]
                ),
                "random_state": 42,
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)

        elif model_name == "Ridge":
            params = {
                "alpha": trial.suggest_float(
                    "alpha", 1e-4, 100.0, log=True
                )
            }
            model = Ridge(**params)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        fold_rmse = []

        for train_idx, val_idx in cv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_val)

            rmse = np.sqrt(
                mean_squared_error(y_fold_val, predictions)
            )
            fold_rmse.append(rmse)

        return float(np.mean(fold_rmse))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
    )

    print(f"\n{model_name} best CV RMSE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")

    # Rebuild the model using the best parameters.
    best_params = study.best_params

    if model_name == "ExtraTrees":
        best_model = ExtraTreesRegressor(
            **best_params, random_state=42, n_jobs=1
        )
    elif model_name == "RandomForest":
        best_model = RandomForestRegressor(
            **best_params, random_state=42, n_jobs=1
        )
    else:
        best_model = Ridge(**best_params)

    best_model.fit(X_train, y_train)

    return best_model, study

def build_model(model_name: str):
    """Create a fresh sklearn pipeline; preprocessing is fitted inside each CV fold."""
    if model_name == "Ridge":
        estimator = Ridge()
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    elif model_name == "SVR_RBF":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]
    elif model_name == "RandomForest":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
        ]
    elif model_name == "ExtraTrees":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(random_state=42, n_jobs=1)),
        ]
    elif model_name == "HistGradientBoosting":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps)


def direction_accuracy(y_true, y_pred, previous_close) -> float:
    actual_direction = np.sign(np.asarray(y_true) - np.asarray(previous_close))
    predicted_direction = np.sign(np.asarray(y_pred) - np.asarray(previous_close))
    return float(np.mean(actual_direction == predicted_direction))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Tune and evaluate stock models using chronological validation."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--experiment", default="stock-prediction-model-selection")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_csv:
        csv_path = Path(args.data_csv)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
            if not csv_path.exists():
                csv_path = (project_root / args.data_csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        raw = load_csv_for_ticker(csv_path, args.ticker)
    else:
        csv_path = Path(__file__).resolve().parent / "stock_data.csv"
        raw = load_csv_for_ticker(csv_path, args.ticker)
        
    # make_dataset currently returns nine columns, including Close. Keep Close
    # separately for baseline/direction metrics; only the eight approved columns
    # are supplied to the model.
    X_all, y = make_dataset(raw)
    missing = sorted(set(MODEL_FEATURES) - set(X_all.columns))
    if missing:
        raise ValueError(f"Training dataset is missing required features: {missing}")
    previous_close = X_all["Close"].copy() if "Close" in X_all.columns else None
    X = X_all[MODEL_FEATURES].copy()

    if previous_close is None:
        raise ValueError(
            "make_dataset() must return a Close column so direction accuracy and "
            "the naive baseline can be calculated separately from model features."
        )

    # Keep chronological order and reserve the most recent 20% as final holdout.
    test_start = int(len(X) * 0.8)
    X_dev, X_test = X.iloc[:test_start], X.iloc[test_start:]
    y_dev, y_test = y.iloc[:test_start], y.iloc[test_start:]
    close_test = previous_close.iloc[test_start:]

    if len(X_dev) < 100 or len(X_test) < 20:
        raise ValueError(
            f"Not enough usable data: {len(X_dev)} development rows and "
            f"{len(X_test)} holdout rows. Use a longer history."
        )

    profile = profile_data(X_dev, y_dev)
    selected_votes, vote_reasons = vote_for_models(profile)
    n_splits = min(args.cv_splits, max(2, len(X_dev) // 40))
    cv = TimeSeriesSplit(n_splits=n_splits)

    print(f"Ticker: {args.ticker.upper()}")
    print(f"Rows: {len(X)} | Development: {len(X_dev)} | Holdout: {len(X_test)}")
    print(f"Model features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
    print("\nCandidate models selected by voting:")
    for name, votes in selected_votes.items():
        print(f"  {name}: {votes} votes")

    # MLflow is optional: evaluation still works without it installed.
    mlflow = None
    try:
        import mlflow as mlflow_module
        import mlflow.sklearn as mlflow_sklearn_module
        mlflow = mlflow_module
        mlflow_sklearn = mlflow_sklearn_module
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
    except ImportError:
        print("\nMLflow is not installed; continuing without experiment tracking.")
        mlflow_sklearn = None

    results: dict[str, dict[str, Any]] = {}
    tuned_estimators = {}

    for model_name, vote_count in selected_votes.items():
        print(f"\nTuning {model_name}...")
    
    
        best_model, study = tune_with_optuna(
        model_name="ExtraTrees",
        X_train=X_dev,
        y_train=y_dev,
        n_trials=30,
    )
        predictions = best_model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        dir_acc = direction_accuracy(y_test.to_numpy(), predictions, close_test.to_numpy())
        baseline_predictions = close_test.to_numpy()
        baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_predictions)))

        cv_results_path = output_dir / f"{model_name}_cv_results.csv"
        study.trials_dataframe().to_csv(cv_results_path, index=False)

        result = {
            "voting_score": int(vote_count),
            "best_params": study.best_params,
            "cv_best_rmse": float(study.best_value),
            "holdout_mae": mae,
            "holdout_rmse": rmse,
            "holdout_direction_accuracy": dir_acc,
            "naive_baseline_holdout_mae": baseline_mae,
            "naive_baseline_holdout_rmse": baseline_rmse,
            "mae_improvement_vs_baseline": float(baseline_mae - mae),
            "cv_results_csv": str(cv_results_path),
        }
        results[model_name] = result
        tuned_estimators[model_name] = best_model

        print(
    f"  CV RMSE={study.best_value:.4f} | "
    f"Holdout MAE={mae:.4f} | "
    f"RMSE={rmse:.4f} | "
    f"Direction accuracy={dir_acc:.3f}"
)
    print(f"  Best params: {study.best_params}")
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{args.ticker.upper()}-{model_name}"):
            mlflow.log_params({
                "ticker": args.ticker.upper(),
                "period": args.period,
                "model_name": model_name,
                "voting_score": int(vote_count),
                "cv_strategy": "TimeSeriesSplit",
                "cv_splits": n_splits,
                "feature_count": len(MODEL_FEATURES),
                "development_rows": len(X_dev),
                "holdout_rows": len(X_test),
                **{f"best_{k}": v for k, v in study.best_params.items()},
            })
            mlflow.log_metrics({
                "cv_best_rmse": float(study.best_value),
                "holdout_mae": mae,
                "holdout_rmse": rmse,
                "holdout_direction_accuracy": dir_acc,
                "naive_baseline_holdout_mae": baseline_mae,
                "naive_baseline_holdout_rmse": baseline_rmse,
                "mae_improvement_vs_baseline": baseline_mae - mae,
            })
            mlflow.log_artifact(str(cv_results_path), artifact_path="tuning")
            mlflow.log_dict(result, "metrics.json")
            mlflow_sklearn.log_model(best_model, artifact_path="model")

    # Choose by development-only cross-validation score, not holdout metrics.
    winner_name = min(results, key=lambda name: results[name]["cv_best_rmse"])
    winner = tuned_estimators[winner_name]

    # Refit the chosen configuration on all labelled data for deployment.
    # The holdout metrics above remain the performance estimate from the pre-refit model.
    winner.fit(X, y)

    model_path = output_dir / "stock_model.joblib"
    joblib.dump(winner, model_path)

    report = {
        "ticker": args.ticker.upper(),
        "period": args.period,
        "target": "next_trading_day_close",
        "selected_models_by_vote": selected_votes,
        "vote_reasons": vote_reasons,
        "data_profile": asdict(profile),
        "model_results": results,
        "winner_by_cv_rmse": winner_name,
        "saved_model_path": str(model_path),
        "feature_columns": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "training_rows": int(len(X)),
        "development_rows": int(len(X_dev)),
        "holdout_rows": int(len(X_test)),
        "cv_strategy": "TimeSeriesSplit",
        "cv_splits": n_splits,
        "mlflow_experiment": args.experiment if mlflow is not None else None,
        "note": (
            "Model family and hyperparameters are selected using development-only "
            "TimeSeriesSplit CV RMSE. The final 20% is used for holdout metrics and is "
            "not used for tuning. Since multiple candidates are compared on this holdout, "
            "those metrics are still model-selection evidence rather than a fully "
            "unbiased final estimate. The saved winner is refit on all labelled rows."
        ),
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{args.ticker.upper()}-evaluation-summary"):
            mlflow.log_param("ticker", args.ticker.upper())
            mlflow.log_param("winner", winner_name)
            mlflow.log_param("winner_selection_metric", "cv_best_rmse")
            mlflow.log_metric("winner_cv_best_rmse", results[winner_name]["cv_best_rmse"])
            mlflow.log_metric("winner_holdout_mae", results[winner_name]["holdout_mae"])
            mlflow.log_metric("winner_holdout_rmse", results[winner_name]["holdout_rmse"])
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(model_path))

    print("\n=== Results (ranked by CV RMSE) ===")
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["cv_best_rmse"]):
        print(
            f"{name}: CV RMSE={metrics['cv_best_rmse']:.4f}, "
            f"Holdout MAE={metrics['holdout_mae']:.4f}, "
            f"Holdout RMSE={metrics['holdout_rmse']:.4f}, "
            f"Baseline RMSE={metrics['naive_baseline_holdout_rmse']:.4f}"
        )
    print(f"\nWinner by CV RMSE: {winner_name}")
    print(f"Saved model: {model_path}")
    print(f"Evaluation report: {report_path}")
    if mlflow is not None:
        print(f"MLflow experiment: {args.experiment}")
        print("Start the MLflow UI with: mlflow ui")


if __name__ == "__main__":
    main()
