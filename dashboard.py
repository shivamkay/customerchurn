from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px




from preprocess import _engineer_rfm, _merge_customer_features, simulate_churn_if_missing
from utils import safe_parse_dates, robust_fillna
from models import train_supervised_models, segment_customers, predict_proba
from sales_analysis import compute_monthly_sales, compute_periodic_sales, top_products

st.set_page_config(page_title="Customer Churn Prediction & Sales Dashboard", layout="wide")
st.title("Customer Churn Prediction & Sales Dashboard")
st.caption("Upload CSVs or place them in `data/` and hit 'Load sample/Local data'.")

with st.sidebar:
    st.header("Data Sources")
    cfile = st.file_uploader("customers.csv", type=["csv"])
    tfile = st.file_uploader("transactions.csv", type=["csv"])
    pfile = st.file_uploader("products.csv (optional)", type=["csv"])
    load_local = st.button("Load Local data (from ./data/)")

def read_uploaded_or_local(cfile, tfile, pfile):
    import os
    if cfile and tfile:
        customers = pd.read_csv(cfile)
        transactions = pd.read_csv(tfile)
        products = pd.read_csv(pfile) if pfile else None
        return customers, transactions, products
    cpath, tpath, ppath = "data/customers.csv", "data/transactions.csv", "data/products.csv"
    if all([os.path.exists(cpath), os.path.exists(tpath)]):
        return pd.read_csv(cpath), pd.read_csv(tpath), (pd.read_csv(ppath) if os.path.exists(ppath) else None)
    st.stop()

if load_local or (cfile and tfile):
    customers, transactions, products = read_uploaded_or_local(cfile, tfile, pfile)

    customers = robust_fillna(safe_parse_dates(customers, ["signup_date"]))
    transactions = safe_parse_dates(transactions, ["date"])
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)

    rfm = _engineer_rfm(transactions)
    features = _merge_customer_features(customers, rfm)

    if "churned" in customers.columns:
        labels = customers.set_index("customer_id")["churned"]
    else:
        labels = simulate_churn_if_missing(features)
        labels = pd.Series(labels, index=features["customer_id"], name="churned")

    drop_cols = ["customer_id","last_purchase","first_purchase","signup_date"]
    cat_cols = [c for c in features.columns if features[c].dtype=="object" and c not in drop_cols]




    features_encoded = pd.get_dummies(features, columns=cat_cols, drop_first=True)\
                        .replace([np.inf, -np.inf], np.nan)\
                        .fillna(0)

    # Ensure customer_id is a column and not duplicated
    if "customer_id" not in features_encoded.columns:
        features_encoded = features_encoded.reset_index()
    labels = labels.reindex(features_encoded["customer_id"]).fillna(0).astype(int)

    st.success(f"Loaded {len(customers)} customers, {len(transactions)} transactions.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Churn Modeling", "Sales Trends", "Segmentation", "Download"])

    with tab1:
        st.subheader("Training & Evaluation")
        if len(labels.unique()) < 2:
            st.warning(
                "⚠️ Dataset has only one class. Model training requires at least 2 classes."
            )
        else:
            X_model = features_encoded.drop(drop_cols, axis=1, errors="ignore").apply(pd.to_numeric)
            train_out, metrics = train_supervised_models(X_model, labels)
            st.json({k: {"roc_auc": round(v["roc_auc"],4)} for k,v in metrics.items()})
            best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
            st.info(f"Best model: **{best_name}** (ROC-AUC={metrics[best_name]['roc_auc']:.3f})")
            best_model = train_out["models"][best_name]
            proba_df = predict_proba(best_model, train_out["scaler"], X_model)
            st.dataframe(proba_df.sort_values("churn_probability", ascending=False).head(20), use_container_width=True)

            st.subheader("Confusion Matrix (best model @ 0.5)")
            from sklearn.metrics import confusion_matrix
            X_test, y_test = train_out["X_test"], train_out["y_test"]
            y_prob = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
            st.dataframe(cm_df)

    with tab2:
        st.subheader("Sales Trends")
        m = compute_monthly_sales(transactions)
        q = compute_periodic_sales(transactions, "Q")
        y = compute_periodic_sales(transactions, "Y")
        st.plotly_chart(px.line(m, x="month", y="sales", title="Monthly Sales"), use_container_width=True)
        st.plotly_chart(px.bar(q, x="period", y="sales", title="Quarterly Sales"), use_container_width=True)
        st.plotly_chart(px.line(y, x="period", y="sales", title="Yearly Sales"), use_container_width=True)
        st.subheader("Top Products / Categories")
        tp = top_products(transactions, products if products is not None else None, top_n=10)
        st.dataframe(tp, use_container_width=True)

    with tab3:
        st.subheader("Customer Segmentation")
        seg = segment_customers(features_encoded, n_clusters=5)
        merged = features_encoded.merge(seg, on="customer_id", how="left")
        xcol = "recency_days" if "recency_days" in merged.columns else merged.select_dtypes(np.number).columns[0]
        ycol = "monetary" if "monetary" in merged.columns else merged.select_dtypes(np.number).columns[1]
        fig = px.scatter(merged, x=xcol, y=ycol, color="segment", hover_data=["customer_id"])
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Downloads")
        if 'proba_df' in locals():
            st.download_button(
                "Download churn probabilities (CSV)",
                data=proba_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_probabilities.csv",
                mime="text/csv"
            )
else:
    st.info("👈 Upload your CSVs or click Load Local data to auto-load from `data/`.")
