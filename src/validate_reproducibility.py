import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.project_utils import ROOT, PROCESSED_DATA, ARTIFACT_DIR


def main():
    if not PROCESSED_DATA.exists():
        from src.preprocess import preprocess
        preprocess()
    df = pd.read_csv(PROCESSED_DATA).dropna(subset=["Rating"]).copy()
    X = df.drop(columns=["Rating"])
    y = df["Rating"]
    first = train_test_split(X.index, test_size=0.20, random_state=42)
    second = train_test_split(X.index, test_size=0.20, random_state=42)
    reproducible = all(a.equals(b) for a, b in zip(first, second))
    report = {
        "test_name": "Deterministic train/test split validation",
        "random_state": 42,
        "test_size": 0.20,
        "train_rows": len(first[0]),
        "test_rows": len(first[1]),
        "is_reproducible": bool(reproducible),
        "status": "PASSED" if reproducible else "FAILED",
        "scope_note": "This validates split reproducibility, not full model reproducibility.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {report_path}")
    if not reproducible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
