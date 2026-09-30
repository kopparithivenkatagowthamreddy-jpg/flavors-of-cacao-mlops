import json
import mlflow
from mlflow import MlflowClient
from src.project_utils import ROOT, MLFLOW_DB, MODEL_NAME, ARTIFACT_DIR


def _r2_score(client, model_version):
    if not model_version.run_id:
        return None
    run = client.get_run(model_version.run_id)
    for key in ("R2", "r2", "r2_score"):
        if key in run.data.metrics:
            return float(run.data.metrics[key])
    return None


def automate_champion_challenger():
    print("[INFO] Starting Flavors of Cacao model lifecycle manager...")
    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as exc:
        raise RuntimeError(
            f"Could not query model '{MODEL_NAME}'. Run src.train_registry first."
        ) from exc

    if not versions:
        print("[INFO] No registered versions found. Run src.train_registry first.")
        return

    # MLflow model stages are deprecated in newer releases, so use version tags
    # for lifecycle labels while retaining version metrics from MLflow runs.
    candidates = []
    for version in versions:
        score = _r2_score(client, version)
        if score is None:
            print(f"[WARNING] Version {version.version} has no R2 metric; skipping.")
            continue
        candidates.append((version, score))
        client.set_model_version_tag(
            MODEL_NAME, str(version.version), "lifecycle_stage",
            (version.tags or {}).get("lifecycle_stage", "candidate")
        )
        print(f"[INFO] Version {version.version} | R2={score:.4f}")

    if not candidates:
        print("[ERROR] No registered model versions have an R2 metric.")
        return

    # Preserve the current production-tagged version unless a candidate has a
    # strictly higher R2 score. If none is tagged production, select the top R2.
    production = [
        (v, score) for v, score in candidates
        if (v.tags or {}).get("lifecycle_stage", "").lower() == "production"
    ]
    champion = max(production, key=lambda item: item[1]) if production else None
    best = max(candidates, key=lambda item: item[1])

    if champion is None or best[1] > champion[1]:
        chosen, chosen_score = best
        client.set_model_version_tag(MODEL_NAME, str(chosen.version), "lifecycle_stage", "production")
        client.set_model_version_tag(MODEL_NAME, str(chosen.version), "validation_status", "metric_compared")
        if champion and champion[0].version != chosen.version:
            client.set_model_version_tag(MODEL_NAME, str(champion[0].version), "lifecycle_stage", "archived")
        action = "production_tag_updated"
        print(f"[INFO] Version {chosen.version} tagged production (R2={chosen_score:.4f}).")
    else:
        chosen, chosen_score = champion
        action = "existing_production_retained"
        print(f"[INFO] Existing production-tagged version {chosen.version} retained.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": MODEL_NAME,
        "metric_used": "R2 (higher is better for this comparison)",
        "action": action,
        "selected_version": str(chosen.version),
        "selected_r2": chosen_score,
        "versions_checked": len(candidates),
        "note": "Lifecycle labels are MLflow model-version tags, not deprecated MLflow stages. Metric comparison is not a complete deployment validation.",
    }
    (ARTIFACT_DIR / "lifecycle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[SUCCESS] Lifecycle management completed.")


if __name__ == "__main__":
    automate_champion_challenger()
