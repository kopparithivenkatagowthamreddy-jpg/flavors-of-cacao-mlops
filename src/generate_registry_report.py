import json
import mlflow
from mlflow import MlflowClient
from src.project_utils import ROOT, MLFLOW_DB, MODEL_NAME, ARTIFACT_DIR


def generate_registry_report():
    print("[INFO] Generating Flavors of Cacao registry report...")
    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    registered = client.get_registered_model(MODEL_NAME)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    version_reports = []
    for version in sorted(versions, key=lambda item: int(item.version)):
        metrics = {}
        params = {}
        if version.run_id:
            run = client.get_run(version.run_id)
            metrics = dict(run.data.metrics)
            params = dict(run.data.params)
        version_reports.append({
            "version": str(version.version),
            "run_id": version.run_id,
            "status": version.status,
            "lifecycle_stage": (version.tags or {}).get("lifecycle_stage", "not_set"),
            "validation_status": (version.tags or {}).get("validation_status", "not_set"),
            "metrics": metrics,
            "parameters": params,
            "source": version.source,
        })

    report = {
        "project": "Flavors of Cacao Rating Prediction",
        "model_name": MODEL_NAME,
        "description": registered.description,
        "registered_versions_count": len(version_reports),
        "versions": version_reports,
        "generated_note": "This report summarizes registry metadata and logged metrics; it does not independently certify production readiness.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / "model_registry_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SUCCESS] Registry report saved to: {report_path}")


if __name__ == "__main__":
    generate_registry_report()
