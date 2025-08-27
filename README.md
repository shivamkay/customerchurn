# Customer Churn Prediction and Sales Dashboard

A complete, ready-to-run project that ingests customer + transaction data, engineers features,
trains churn prediction models, segments customers, analyzes sales trends, and serves an interactive dashboard.

---

## 📦 Project Structure
```
churn_sales_project/
├─ data/                         # put your CSVs here (or upload via dashboard)
│  ├─ customers.csv              # required: customer_id, signup_date, churned(optional), ...
│  ├─ transactions.csv           # required: transaction_id, customer_id, date, amount, product_id(optional)
│  └─ products.csv               # optional: product_id, category, name, ...
├─ src/
│  ├─ preprocess.py              # ETL + feature engineering
│  ├─ models.py                  # churn models, segmentation
│  ├─ sales_analysis.py          # sales trend KPIs
│  ├─ dashboard.py               # Streamlit interactive app
│  └─ utils.py                   # shared helpers
├─ requirements.txt
└─ README.md
```

> Works even if `products.csv` or a `churned` column are missing; the app will guide you and/or simulate labels for demo.

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
Place your files in `data/` with **these minimal columns**:

- `customers.csv`: `customer_id`, `signup_date`, *(optional)* `churned` (0/1), any profile fields.
- `transactions.csv`: `transaction_id`, `customer_id`, `date`, `amount`, *(optional)* `product_id`.
- `products.csv` *(optional)*: `product_id`, `category`, `name`.

Dates can be `YYYY-MM-DD` or ISO 8601.

### 3) Run the Dashboard
```bash
streamlit run src/dashboard.py
```
Upload your CSVs in-app **or** it will auto-load from `data/` if found.

### 4) CLI (optional)
You can also call modules directly, e.g. from a Python shell or notebook:
```python
from src.preprocess import load_data_and_engineer
from src.models import train_supervised_models, segment_customers
from src.sales_analysis import compute_monthly_sales

dfs = load_data_and_engineer("data/customers.csv","data/transactions.csv","data/products.csv")
models, metrics = train_supervised_models(dfs["features"], dfs["labels"])
segments = segment_customers(dfs["features"])
msales = compute_monthly_sales(dfs["transactions"])
```

---

## 🧠 Models Included
- **Supervised:** Logistic Regression, Random Forest, Gradient Boosting (sklearn). XGBoost/LightGBM if available.
- **Deep Learning (optional):** Simple Keras MLP if TensorFlow is installed.
- **Unsupervised:** K-Means (segmentation). DBSCAN for anomaly detection (optional toggle).

### Core Features Engineered
- **RFM:** Recency, Frequency, Monetary from transactions.
- **Tenure:** Days since signup.
- **Engagement:** Derived from purchase frequency and recency (and `interactions` if present).
- **AOV:** Average order value, spend velocity.

---

## 📊 Dashboard Highlights
- Upload or auto-load CSVs
- Churn training + evaluation (ROC-AUC, F1, Precision/Recall, Confusion Matrix)
- Customer-level churn probabilities (downloadable CSV)
- Sales trends (monthly/quarterly/yearly)
- Top products/categories
- Segmentation scatter & cohort insights
- Rev impact vs. churn propensity

---

## 📝 Notes
- If no `churned` label is found, the app can **simulate labels** by thresholding recency + inactivity for demo use.
- Handles missing values, encodes categoricals, scales numeric features.
- All charts are interactive (Plotly).

---

## 🔒 Data & Privacy
All processing is local to your environment. No data is sent externally.

---

## 🧾 License
MIT
