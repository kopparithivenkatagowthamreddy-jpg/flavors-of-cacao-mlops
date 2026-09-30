from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "flavors_of_cacao.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "processed_flavors_of_cacao.csv"
MODEL_PATH = ROOT / "models" / "random_forest_regressor.joblib"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"
LOG_DIR = ROOT / "logs"
MLFLOW_DB = ROOT / "mlflow.db"
MODEL_NAME = "Flavors_of_Cacao_Rating_Model"


def load_raw_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA}. Place flavors_of_cacao.csv in data/raw/."
        )
    return pd.read_csv(RAW_DATA)


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    rename_map = {
        "Company (Maker-if known)": "Company",
        "Specific Bean Origin or Bar Name": "Bean_Origin_Bar",
        "Review Date": "Review_Year",
        "Cocoa Percent": "Cocoa_Percent",
        "Company Location": "Company_Location",
        "Bean Type": "Bean_Type",
        "Broad Bean Origin": "Broad_Bean_Origin",
    }
    return df.rename(columns=rename_map)


def prepare_data(df):
    df = clean_column_names(df)
    if "Cocoa_Percent" in df.columns:
        cocoa = (
            df["Cocoa_Percent"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace({"": float("nan"), "nan": float("nan"), "None": float("nan")})
        )
        df["Cocoa_Percent"] = pd.to_numeric(cocoa, errors="coerce")
    for col in ["REF", "Review_Year", "Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
