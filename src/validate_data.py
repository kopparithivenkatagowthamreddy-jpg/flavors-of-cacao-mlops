from src.project_utils import load_raw_data, prepare_data


def validate():
    raw = load_raw_data()
    df = prepare_data(raw)
    required = {"Rating", "Cocoa_Percent", "Review_Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df["Rating"].dropna().empty:
        raise ValueError("Rating target contains no valid numeric values.")
    if df["Rating"].dropna().lt(0).any() or df["Rating"].dropna().gt(5).any():
        raise ValueError("Rating values outside expected 0–5 range were found.")
    print("[SUCCESS] Flavors of Cacao data validation passed.")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Missing values total: {int(df.isna().sum().sum())}")
    return True


if __name__ == "__main__":
    validate()
