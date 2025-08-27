from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN

def _xy(df_features: pd.DataFrame, y: pd.Series):
    X = df_features.drop(columns=[c for c in ["churned"] if c in df_features.columns])
    y = y.loc[X.index]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, y, scaler

def _fit_if_available(model_ctor, **kwargs):
    try:
        model = model_ctor(**kwargs)
        return model
    except Exception:
        return None

def train_supervised_models(df_features: pd.DataFrame, y: pd.Series, test_size: float=0.2, random_state: int=42):
    X, y, scaler = _xy(df_features, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    results = {}
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, n_jobs=None)
    lr.fit(X_train, y_train)
    models["logreg"] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X_train, y_train)
    models["rf"] = rf

    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)
    models["gboost"] = gb

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    except Exception:
        pass

    # Metrics
    for name, mdl in models.items():
        proba = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else mdl.decision_function(X_test)
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "report": classification_report(y_test, pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }

    return {"models": models, "scaler": scaler, "X_test": X_test, "y_test": y_test}, results

def segment_customers(df_features: pd.DataFrame, n_clusters: int=5, random_state: int=42) -> pd.DataFrame:
    X = df_features.select_dtypes(include=[np.number]).copy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    df_features = df_features.copy()
    df_features["segment"] = km.fit_predict(X)
    return df_features[["customer_id","segment"]]

def detect_anomalies(df_features: pd.DataFrame, eps: float=0.5, min_samples: int=5) -> pd.DataFrame:
    X = df_features.select_dtypes(include=[np.number]).copy()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    outliers = df_features[labels == -1]
    return outliers[["customer_id"]].assign(anomaly=1)

def predict_proba(model, scaler, df_features: pd.DataFrame) -> pd.DataFrame:
    X = df_features.drop(columns=[c for c in ["churned"] if c in df_features.columns]).copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X[num_cols] = scaler.transform(X[num_cols])
    proba = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return pd.DataFrame({"customer_id": df_features["customer_id"], "churn_probability": proba})
