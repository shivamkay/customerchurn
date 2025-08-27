from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils import safe_parse_dates, ensure_columns, robust_fillna

def _load_csv_maybe(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def _engineer_rfm(transactions: pd.DataFrame, ref_date: Optional[pd.Timestamp]=None) -> pd.DataFrame:
    if ref_date is None:
        ref_date = transactions["date"].max()
    agg = transactions.groupby("customer_id").agg(
        last_purchase=("date","max"),
        frequency=("transaction_id","nunique"),
        monetary=("amount","sum"),
        aov=("amount","mean"),
        first_purchase=("date","min")
    ).reset_index()
    agg["recency_days"] = (ref_date - agg["last_purchase"]).dt.days.clip(lower=0)
    agg["tenure_days"] = (ref_date - agg["first_purchase"]).dt.days.clip(lower=0)
    agg["spend_velocity"] = agg["monetary"] / (agg["tenure_days"].replace(0,1))
    return agg

def _merge_customer_features(customers: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(rfm, on="customer_id", how="left")
    freq_norm = (df["frequency"].fillna(0) / df["frequency"].fillna(0).max()).fillna(0)
    rec_norm = 1 - (df["recency_days"].fillna(df["recency_days"].max()) / df["recency_days"].fillna(0).max()).fillna(0)
    df["engagement_score"] = (0.6 * freq_norm + 0.4 * rec_norm).fillna(0)
    return df

def simulate_churn_if_missing(features: pd.DataFrame) -> pd.Series:
    # Heuristic: churn if no purchase in > 120 days and low engagement
    s = ((features["recency_days"].fillna(999) > 120) & (features["engagement_score"].fillna(0) < 0.3)).astype(int)
    return s.rename("churned")

def load_data_and_engineer(customers_path: str, transactions_path: str, products_path: Optional[str]=None) -> Dict[str, pd.DataFrame]:
    customers = _load_csv_maybe(customers_path)
    transactions = _load_csv_maybe(transactions_path)
    products = _load_csv_maybe(products_path)

    if customers is None or transactions is None:
        raise FileNotFoundError("customers.csv and transactions.csv must exist.")

    customers = ensure_columns(customers, ["customer_id","signup_date"], "customers.csv")
    transactions = ensure_columns(transactions, ["transaction_id","customer_id","date","amount"], "transactions.csv")

    customers = safe_parse_dates(customers, ["signup_date"])
    transactions = safe_parse_dates(transactions, ["date"])
    customers = robust_fillna(customers)
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)

    rfm = _engineer_rfm(transactions)
    features = _merge_customer_features(customers, rfm)

    labels = customers["churned"] if "churned" in customers.columns else simulate_churn_if_missing(features)

    # Categorical encoding (basic one-hot for non-numeric columns, excluding ids/dates)
    drop_cols = ["customer_id","last_purchase","first_purchase","signup_date"]
    cat_cols = [c for c in features.columns if features[c].dtype=="object" and c not in drop_cols]
    features_encoded = pd.get_dummies(features, columns=cat_cols, drop_first=True)

    # Replace inf/NaN
    features_encoded = features_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {
        "customers": customers,
        "transactions": transactions,
        "products": products if products is not None else pd.DataFrame(),
        "features": features_encoded.set_index("customer_id", drop=False),
        "labels": labels.loc[features_encoded["customer_id"]].astype(int),
        "rfm": rfm
    }
