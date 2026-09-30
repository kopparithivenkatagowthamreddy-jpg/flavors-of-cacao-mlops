# Flavors of Cacao Rating Prediction — MLOps Project

This project uses the **same overall folder layout** as the supplied Churn Prediction project, with its dataset, notebook, and supporting scripts adapted for the Flavors of Cacao rating-regression task.

## Important implementation note

The original `241FA18155.ipynb` notebook's model implementation and analysis steps are preserved. The only notebook code change is the absolute local dataset path, replaced with the project-relative path `data/raw/flavors_of_cacao.csv` so it can run on another computer. Notebook outputs were cleared so you can run it from top to bottom.

## Folder structure

```text
flavors-of-cacao-mlops/
├── 241FA18155.ipynb
├── data/
│   ├── raw/
│   │   └── flavors_of_cacao.csv
│   └── processed/
│       └── dataset_metadata.json
├── notebooks/
│   └── 241FA18155.ipynb
├── src/
│   ├── __init__.py
│   ├── project_utils.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess_lab4.py
│   ├── train_mlflow.py
│   ├── evaluate_lab4.py
│   ├── validate_reproducibility.py
│   ├── validate_data.py
│   ├── preprocess_pipeline.py
│   ├── validate_outputs.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   └── generate_registry_report.py
├── pipelines/
│   ├── __init__.py
│   ├── run_baseline.py
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
├── models/
├── outputs/
├── logs/
├── artifacts/
├── requirements.txt
└── .gitignore
```

## Setup

Run commands from the project root.

### 1. Create and activate a virtual environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Open the original notebook

```powershell
jupyter notebook notebooks/241FA18155.ipynb
```

Run all cells from top to bottom. The notebook predicts `Rating`, a continuous numeric target, so it is a **regression** project.

## Supporting MLOps scripts

These scripts are additional project infrastructure and do not replace the notebook implementation.

```powershell
python -m src.validate_data
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

Or run the complete baseline workflow:

```powershell
python pipelines/run_baseline.py
```

Optional experiment tracking:

```powershell
python pipelines/run_lab4_tracking.py
```

MLflow tracking requires an accessible MLflow tracking server for persistent team tracking/registry workflows. The baseline workflow does not require a server.

## Main project details

- Dataset: `data/raw/flavors_of_cacao.csv`
- Target: `Rating`
- Problem type: Regression
- Notebook: `notebooks/241FA18155.ipynb`
- Supporting saved model: `models/random_forest_regressor.joblib`
- Supporting evaluation report: `outputs/regression_metrics.json`

The notebook itself saves its model using the filename defined in its original code. The supporting scripts use a separate model path so they do not overwrite the notebook's model.


## Train and save all five models

The project includes `src/train_all_models.py`, which trains and saves Linear Regression,
Ridge Regression, Random Forest, Tuned Random Forest, and Gradient Boosting pipelines.

Run from the project root:

```powershell
python -m src.train_all_models
```

The script saves fitted pipelines under `models/`, writes freshly measured metrics to
`outputs/model_comparison_latest.csv`, and stores tuning/training details under `artifacts/`.
The `outputs/model_comparison.csv` file preserves the five model metrics supplied from the
notebook results. Freshly trained models may have slightly different metrics depending on
the tuning run and installed library versions.
