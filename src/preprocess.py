import json
from src.project_utils import (
    ROOT, RAW_DATA, PROCESSED_DATA, ARTIFACT_DIR,
    prepare_data, load_raw_data,
)


def preprocess():
    raw = load_raw_data()
    cleaned = prepare_data(raw)
    rows_before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA, index=False)

    metadata = {
        "dataset": str(RAW_DATA.relative_to(ROOT)),
        "processed_dataset": str(PROCESSED_DATA.relative_to(ROOT)),
        "raw_shape": list(raw.shape),
        "processed_shape": list(cleaned.shape),
        "duplicate_rows_removed": int(rows_before_deduplication - len(cleaned)),
        "target": "Rating",
        "problem_type": "regression",
        "preprocessing_steps": [
            "clean column names", "convert Cocoa Percent to numeric",
            "convert REF, Review Date, and Rating to numeric",
            "remove duplicate rows",
        ],
    }
    metadata_path = PROCESSED_DATA.parent / "dataset_metadata.json"
    metadata_text = json.dumps(metadata, indent=2)
    metadata_path.write_text(metadata_text, encoding="utf-8")
    (ARTIFACT_DIR / "preprocessing_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )
    return cleaned


if __name__ == "__main__":
    data = preprocess()
    print("[SUCCESS] Preprocessing completed.")
    print(f"[INFO] Saved: {PROCESSED_DATA}")
    print(f"[INFO] Shape: {data.shape}")
