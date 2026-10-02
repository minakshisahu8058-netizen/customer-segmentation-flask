# Customer Segmentation Analysis (RFM + K-Means + Flask)

Segments retail customers using RFM analysis and predicts a customer's segment through a Flask web app.

## Tools
Python, Pandas, NumPy, Scikit-learn, Flask, Matplotlib, Seaborn

## What I did
- Cleaned 500K+ retail transaction records (missing values, duplicates, cancelled orders)
- Built Recency, Frequency and Monetary features for each customer
- Applied K-Means clustering and chose the cluster count with the elbow method
- Built a Flask app: enter R, F, M values and get the segment with a suggested action

## Segments
Champions, Loyal Customers, At-Risk Customers, Lost Customers

## How to run
1. pip install -r requirements.txt
2. Download the Online Retail dataset (UCI / Kaggle) and save it as online_retail.csv
3. python train_model.py
4. python app.py
5. Open http://127.0.0.1:5000
