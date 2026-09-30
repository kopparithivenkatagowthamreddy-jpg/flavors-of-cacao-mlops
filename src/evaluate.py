import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.project_utils import PROCESSED_DATA, MODEL_PATH, OUTPUT_DIR


def evaluate():
    if not PROCESSED_DATA.exists():
        from src.preprocess import preprocess
        preprocess()
    if not MODEL_PATH.exists():
        from src.train import train
        train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA).dropna(subset=["Rating"]).copy()
    X = df.drop(columns=["Rating"])
    y = df["Rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }
    folds = min(5, len(df))
    if folds >= 2:
        cv_scores = cross_val_score(
            model, X, y, cv=folds, scoring="neg_mean_absolute_error", n_jobs=1
        )
        metrics["CV_MAE_mean"] = float((-cv_scores).mean())

    (OUTPUT_DIR / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "actual_rating": y_test.to_numpy(),
        "predicted_rating": predictions,
    }).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
