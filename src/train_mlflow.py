import mlflow
import mlflow.sklearn
from src.project_utils import ROOT, MLFLOW_DB, MODEL_PATH
from src.train import train
from src.evaluate import evaluate

EXPERIMENT_NAME = "Flavors-of-Cacao-Rating"


def configure_mlflow():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def run():
    configure_mlflow()
    with mlflow.start_run(run_name="random-forest-regressor"):
        model, *_ = train()
        metrics = evaluate()
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 300,
            "random_state": 42,
            "test_size": 0.20,
        })
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.sklearn.log_model(
            sk_model=model,
            name="flavors-of-cacao-regressor",
            serialization_format="cloudpickle",
        )
        if MODEL_PATH.exists():
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_files")
        return metrics


if __name__ == "__main__":
    print(run())
