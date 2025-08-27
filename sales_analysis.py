from __future__ import annotations
import pandas as pd
import numpy as np

def compute_monthly_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["month"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    monthly = tx.groupby("month", as_index=False)["amount"].sum().rename(columns={"amount":"sales"})
    return monthly

def compute_periodic_sales(transactions: pd.DataFrame, freq: str="Q") -> pd.DataFrame:
    tx = transactions.copy()
    tx["period"] = tx["date"].dt.to_period(freq).dt.to_timestamp()
    return tx.groupby("period", as_index=False)["amount"].sum().rename(columns={"amount":"sales"})

def top_products(transactions: pd.DataFrame, products: pd.DataFrame | None=None, top_n: int=10) -> pd.DataFrame:
    if "product_id" not in transactions.columns:
        return pd.DataFrame(columns=["product_id","sales"])
    prod_sales = transactions.groupby("product_id", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    prod_sales = prod_sales.rename(columns={"amount":"sales"}).head(top_n)
    if products is not None and not products.empty and "product_id" in products.columns:
        prod_sales = prod_sales.merge(products, on="product_id", how="left")
    return prod_sales

def revenue_impact_of_churn(transactions: pd.DataFrame, churn_probs: pd.DataFrame, cutoff: float=0.5) -> pd.DataFrame:
    tx = transactions.copy()
    churners = churn_probs[churn_probs["churn_probability"] >= cutoff]["customer_id"]
    rev_churn = tx[tx["customer_id"].isin(churners)].groupby("customer_id", as_index=False)["amount"].sum().rename(columns={"amount":"historical_revenue"})
    rev_churn["flag"] = "High-risk group"
    return rev_churn
