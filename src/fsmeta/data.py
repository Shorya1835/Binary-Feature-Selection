from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import Config


def infer_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    cols = set(df.columns)

    if dataset_name == "santander":
        if "target" in cols:
            return "target"
        raise ValueError("For santander, 'target' column not found.")

    for col in ["income", "Income", "target", "Target", "class", "Class", "label"]:
        if col in cols:
            return col

    return df.columns[-1]


def load_census(config: Config):
    paths = [
        "/kaggle/input/datasets/tawfikelmetwally/census-income-dataset/adult.csv",
    ]

    csv_path = next((path for path in paths if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Census CSV not found. Check the dataset path under /kaggle/input/."
        )

    df = pd.read_csv(csv_path)

    if config.sample_rows is not None and len(df) > config.sample_rows:
        df = df.sample(config.sample_rows, random_state=config.random_state).reset_index(drop=True)

    target_col = infer_target_column(df, "census")
    x_df = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    for col in x_df.select_dtypes(include=["object"]).columns:
        x_df[col] = x_df[col].astype(str).str.strip()

    if y.dtype == "object":
        y = y.astype(str).str.strip()

    x_df = pd.get_dummies(x_df, drop_first=False)

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.values

    x = x_df.values.astype(float)
    feature_names = x_df.columns.tolist()
    return x, y, feature_names, df


def load_santander(config: Config):
    paths = [
        "/kaggle/input/competitions/santander-customer-transaction-prediction/train.csv",
    ]

    csv_path = next((path for path in paths if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Santander train.csv not found. Check the dataset path under /kaggle/input/."
        )

    df = pd.read_csv(csv_path)

    if config.sample_rows is not None and len(df) > config.sample_rows:
        df = df.sample(config.sample_rows, random_state=config.random_state).reset_index(drop=True)

    target_col = infer_target_column(df, "santander")
    drop_cols = [target_col]
    if "ID_code" in df.columns:
        drop_cols.append("ID_code")

    x_df = df.drop(columns=drop_cols).copy()
    y = df[target_col].values
    x = x_df.values.astype(float)
    feature_names = x_df.columns.tolist()
    return x, y, feature_names, df


def load_dataset(config: Config):
    if config.dataset_name == "census":
        return load_census(config)
    if config.dataset_name == "santander":
        return load_santander(config)
    raise ValueError("dataset_name must be 'census' or 'santander'")


def make_split(x, y, config: Config):
    return train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
