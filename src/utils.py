"""Shared helpers for the Explainable-AI Customer Churn project.

Every notebook imports from here so that the data split, preprocessing and
SHAP handling are identical everywhere (reproducibility).
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
VIS_DIR = ROOT / "visualizations"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
for _d in (DATA_DIR, VIS_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Load the raw Telco CSV. Falls back to downloading with kagglehub."""
    if not DATA_PATH.exists():
        try:
            import kagglehub

            path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
            shutil.copy(path / DATA_PATH.name, DATA_PATH)
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}.\n"
                "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
                "and place the CSV in the dataset/ folder."
            ) from exc
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning only (no statistics learned from data -> no leakage).

    * drop the customerID identifier
    * TotalCharges is stored as text; blanks (customers with tenure 0) -> NaN
      (imputed later, inside the pipeline, using training data only)
    * SeniorCitizen 0/1 -> "No"/"Yes" so it is treated like the other binary columns
    * Churn Yes/No -> 1/0
    """
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def load_split(test_size: float = TEST_SIZE, seed: int = SEED):
    """Deterministic stratified train/test split of the cleaned data."""
    df = clean(load_raw())
    X, y = df.drop(columns="Churn"), df["Churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Median-impute + (optionally) scale numerics, one-hot encode categoricals.

    scale=True  -> Logistic Regression (comparable coefficients, stable solver)
    scale=False -> tree models (scaling is unnecessary; keeps SHAP/LIME feature
                   values in original units such as months and currency)
    Always used inside a Pipeline, so it is fitted on training data only.
    """
    cat_cols = [c for c in X.columns if c not in NUMERIC_COLS]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


# --------------------------------------------------------------------------- #
# SHAP helper (handles different shap versions' output shapes)
# --------------------------------------------------------------------------- #
def shap_positive_class(explainer, X):
    """Return (shap_values[n, p], base_value) for the positive (churn) class."""
    sv = explainer.shap_values(X)
    ev = np.atleast_1d(explainer.expected_value)
    if isinstance(sv, list):          # old API: list per class
        return np.asarray(sv[1]), float(ev[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # new API: (n, p, classes)
        return sv[:, :, 1], float(ev[-1])
    return sv, float(ev[-1])
