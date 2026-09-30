import json
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from src.project_utils import ROOT, MLFLOW_DB, MODEL_NAME, MODEL_PATH
from src.train import train
from src.evaluate import evaluate

EXPERIMENT_NAME = "Flavors-of-Cacao-Registry"


def main():
    print("[INFO] Training and registering Flavors of Cacao rating model...")
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model, *_ = train()
    metrics = evaluate()

    with mlflow.start_run(run_name="cacao-rating-registry-run") as run:
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 300,
            "random_state": 42,
            "test_size": 0.20,
            "target": "Rating",
        })
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="cacao-rating-model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )
        if MODEL_PATH.exists():
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_files")
        run_id = run.info.run_id

    # Apply initial lifecycle metadata to the newly registered version.
    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB.as_posix()}")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [v for v in versions if v.run_id == run_id]
    if matching:
        version = max(matching, key=lambda v: int(v.version))
        client.set_model_version_tag(MODEL_NAME, str(version.version), "lifecycle_stage", "candidate")
        client.set_model_version_tag(MODEL_NAME, str(version.version), "validation_status", "pending")
        print(f"[INFO] Registered version: {version.version}")
    else:
        print("[WARNING] Run completed, but the registered version could not be located by run ID.")

    summary = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "metrics": metrics,
        "model_uri": getattr(model_info, "model_uri", None),
    }
    out = ROOT / "artifacts" / "registry_training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[SUCCESS] Model registration step completed. Summary: {out}")


if __name__ == "__main__":
    main()
