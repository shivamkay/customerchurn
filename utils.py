from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List

def safe_parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to datetime. Invalid parsing becomes NaT.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def ensure_columns(df: pd.DataFrame, required_cols: List[str], name: str) -> pd.DataFrame:
    """
    Ensure the dataframe contains all required columns.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return df

def infer_binary_series(s: pd.Series) -> bool:
    """
    Check if a series contains only 0/1 values (after coercion to numeric).
    """
    try:
        vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        return vals.issubset({0, 1}) and len(vals) > 0
    except Exception:
        return False

def robust_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values safely:
    - numeric columns → 0
    - datetime columns → earliest valid date
    - categorical/object columns → "Unknown"
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(df[col].min())
        else:
            df[col] = df[col].fillna("Unknown")
    return df
