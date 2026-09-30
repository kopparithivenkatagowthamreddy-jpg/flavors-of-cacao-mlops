"""Train, save, and compare all regression models used in the notebook.

Run from the project root:
    python -m src.train_all_models

The model_comparison.csv file stores the user's previously reported metrics.
This script writes freshly measured metrics to model_comparison_latest.csv.
"""
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.project_utils import ROOT, RAW_DATA

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"


def load_notebook_features():
    """Apply the same feature engineering and feature drops as the notebook."""
    df = pd.read_csv(RAW_DATA)
    # Match the notebook's whitespace cleanup without renaming original headers.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Notebook feature engineering
    df["Cocoa_Percent"] = (
        df["Cocoa Percent"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
    )
    df["Cocoa_Percent"] = pd.to_numeric(df["Cocoa_Percent"], errors="coerce")
    df["Review_Year"] = pd.to_numeric(df["Review Date"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    if df.duplicated().any():
        df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Rating", "Cocoa Percent", "Review Date"])
    y = df["Rating"]
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }


def main():
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_notebook_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Each pipeline owns its preprocessor so fitting one model doesn't leak
    # fitted preprocessing state into another model.
    model_specs = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    results = []
    for name, estimator in model_specs.items():
        model = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        results.append({"Model": name.replace("_", " ").title(), **metrics})
        print(f"Saved {name}.joblib | MAE={metrics['MAE']:.4f} "
              f"RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    # Use single-process search to avoid the user's Windows/joblib worker issue.
    # This quick project run uses 5 candidates x 3 folds; the notebook's original
    # search configuration can be run separately if desired.
    tuned_base = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])
    param_distributions = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.5, 0.75, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=tuned_base,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="r2",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        verbose=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_metrics = evaluate(tuned_model, X_test, y_test)
    joblib.dump(tuned_model, MODEL_DIR / "tuned_random_forest.joblib")
    results.append({"Model": "Tuned Random Forest", **tuned_metrics})
    (ARTIFACT_DIR / "tuned_random_forest_params.json").write_text(
        json.dumps(search.best_params_, indent=2), encoding="utf-8"
    )
    print("Saved tuned_random_forest.joblib")
    print("Best tuning parameters:", search.best_params_)

    fresh_metrics = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]]
    fresh_metrics.to_csv(OUTPUT_DIR / "model_comparison_latest.csv", index=False)
    (ARTIFACT_DIR / "model_training_summary.json").write_text(
        json.dumps({
            "dataset": str(RAW_DATA.relative_to(ROOT)),
            "target": "Rating",
            "test_size": 0.20,
            "random_state": 42,
            "tuned_search": "RandomizedSearchCV, 5 candidates, 3-fold CV, n_jobs=1",
            "seconds": round(time.time() - started, 2),
            "saved_models": sorted(p.name for p in MODEL_DIR.glob("*.joblib")),
            "fresh_metrics_file": "outputs/model_comparison_latest.csv",
        }, indent=2), encoding="utf-8"
    )
    print("\nFresh evaluation metrics saved to outputs/model_comparison_latest.csv")
    print(fresh_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
