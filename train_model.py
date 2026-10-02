"""Step 1: Clean data -> build RFM -> train K-Means -> save model files.
Dataset: Online Retail (UCI / Kaggle). Save it here as 'online_retail.csv'.
If the file is missing, a small fake dataset is generated so you can test the app.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

K = 4  # number of segments (pick using the elbow method)

def load_data():
    if os.path.exists("online_retail.csv"):
        return pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
    print("online_retail.csv not found -> using fake demo data")
    rng = np.random.default_rng(42)
    n = 5000
    return pd.DataFrame({
        "InvoiceNo": rng.integers(10000, 20000, n).astype(str),
        "Quantity": rng.integers(1, 12, n),
        "InvoiceDate": pd.to_datetime("2011-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "UnitPrice": rng.uniform(1, 40, n).round(2),
        "CustomerID": rng.integers(12000, 12600, n).astype(float),
    })

# 1. Clean
df = load_data()
df = df.dropna(subset=["CustomerID"])
df = df.drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # remove cancelled orders
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Total"] = df["Quantity"] * df["UnitPrice"]

# 2. RFM
today = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda d: (today - d.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Total", "sum"),
)

# 3. Scale + K-Means
scaler = StandardScaler()
X = scaler.fit_transform(np.log1p(rfm))      # log reduces the effect of huge spenders
model = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["Cluster"] = model.labels_

# 4. Name the clusters (better R/F/M => better segment)
means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
score = (means["Frequency"].rank() + means["Monetary"].rank() - means["Recency"].rank())
order = score.sort_values(ascending=False).index.tolist()
names = ["Champions", "Loyal Customers", "At-Risk Customers", "Lost Customers"]
label_map = {int(c): names[i] for i, c in enumerate(order)}

joblib.dump({"model": model, "scaler": scaler, "labels": label_map}, "segment_model.pkl")
print(means.round(1))
print("Saved segment_model.pkl ->", label_map)
