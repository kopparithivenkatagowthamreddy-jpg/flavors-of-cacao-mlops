import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.project_utils import ROOT, PROCESSED_DATA, MODEL_PATH, ARTIFACT_DIR


def train():
    if not PROCESSED_DATA.exists():
        from src.preprocess import preprocess
        preprocess()

    df = pd.read_csv(PROCESSED_DATA).dropna(subset=["Rating"]).copy()
    if df.empty:
        raise ValueError("No rows with a valid numeric Rating target were found.")

    X = df.drop(columns=["Rating"])
    y = df["Rating"]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=1
        )),
    ])
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    split_info = {
        "test_size": 0.20,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "target": "Rating",
    }
    (ARTIFACT_DIR / "split_data.json").write_text(
        json.dumps(split_info, indent=2), encoding="utf-8"
    )
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train()
    print(f"[SUCCESS] Model saved to: {MODEL_PATH}")
    print(f"[INFO] Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
