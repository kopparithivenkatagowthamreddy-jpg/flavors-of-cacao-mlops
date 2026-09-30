import json
from src.project_utils import ROOT, MODEL_PATH, PROCESSED_DATA, OUTPUT_DIR, ARTIFACT_DIR


def validate_outputs():
    required = [
        PROCESSED_DATA,
        MODEL_PATH,
        OUTPUT_DIR / "regression_metrics.json",
        ARTIFACT_DIR / "split_data.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    report = {
        "status": "FAILED" if missing else "PASSED",
        "required_outputs": [str(path.relative_to(ROOT)) for path in required],
        "missing_outputs": missing,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / "output_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing:
        print("[ERROR] Missing expected outputs:")
        for item in missing:
            print(f" - {item}")
        print(f"Report saved to: {report_path}")
        return False
    print("[SUCCESS] All expected data, model, metrics, and split outputs are present.")
    print(f"Report saved to: {report_path}")
    return True


if __name__ == "__main__":
    if not validate_outputs():
        raise SystemExit(1)
