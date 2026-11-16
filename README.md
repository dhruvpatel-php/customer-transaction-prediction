# 🏦 Customer Transaction Prediction (Santander Banking Benchmark)
### Project ID: PRCP-1003 | Live Streamlit Cloud Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Model-LightGBM%20Gradient%20Boosting-red)](https://lightgbm.readthedocs.io/)
[![ROC-AUC](https://img.shields.io/badge/Holdout%20ROC--AUC-0.8838-success)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An enterprise-grade, interactive Machine Learning application built using **Streamlit** and **LightGBM** to predict customer financial transaction propensity. Designed for retail banking marketing intelligence, customer lifetime value optimization, and high-throughput vector inference.

---

## 🌟 Key Application Features

1. **🔮 Single Customer Propensity Scoring:**
   - 1-click Demo presets for both positive (transaction executed) and negative (no transaction) profiles.
   - Interactive sliders for the **Top 10 most influential features** (`var_34`, `var_174`, `var_76`, `var_6`, etc.).
   - Full 200-variable input drawer with automatic row moment calculation (`row_mean`, `row_std`, `row_min`, `row_max`, `row_sum`).
   - Dynamic threshold sensitivity comparator highlighting standard ($\theta = 0.50$) vs cost-optimal ($\theta^* = 0.22$) decisions.

2. **📁 High-Throughput Batch Scoring (CSV Ingestion):**
   - Upload customer batch CSV files with anonymized variables.
   - Built-in downloadable `sample_batch.csv` test file for immediate evaluation.
   - Real-time KPI summaries: Total Scored, Predicted Transactions, Propensity Rate %.
   - Interactive propensity probability distribution histogram.
   - Download complete scored reports with predicted probabilities and business action tags.

3. **📊 Model Analytics & Diagnostic Dashboard:**
   - Benchmark validation ROC-AUC score (**0.8838**).
   - Top 15 most discriminative features by split gain.
   - Interactive Confusion Matrix comparison demonstrating false negative reduction.
   - Cross-paradigm benchmark table (Logistic Regression vs Random Forest vs LightGBM vs XGBoost).

4. **📑 Comprehensive Technical Architecture & Report:**
   - Full documentation of data challenges, mathematical justifications, and business recommendations.

---

## 🔬 Machine Learning Architecture

- **Dataset Scale:** 200,000 banking instances $\times$ 200 anonymized continuous features (`var_0` to `var_199`).
- **Feature Engineering:** Horizontal statistical moments synthesized across feature vectors:
  $$\text{row\_mean}, \quad \text{row\_std}, \quad \text{row\_min}, \quad \text{row\_max}, \quad \text{row\_sum}$$
- **Champion Algorithm:** Regularized LightGBM with leaf-wise tree growth (`num_leaves=15`, `max_depth=4`, `colsample_bytree=0.4`, `subsample=0.8`).
- **Class Imbalance Handling:** Stratified 5-Fold Cross-Validation with algorithmic cost weighting to preserve authentic distributions without synthetic SMOTE artifacts.
- **Asymmetric Cost Optimization:** Empirical threshold tuning ($\theta^* \approx 0.22$) to overcome the severe 9:1 class skew and minimize high-cost False Negatives in financial customer acquisition.

---

## 🚀 Local Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/dhruvpatel-php/customer-transaction-prediction.git
cd customer-transaction-prediction
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
The app will launch at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment Guide

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **"New app"**.
3. Select your repository: `dhruvpatel-php/customer-transaction-prediction`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your live application within 1-2 minutes!

---

## 📂 Repository Structure

```
├── .streamlit/
│   └── config.toml             # Streamlit visual theme & server configuration
├── app.py                      # Main production Streamlit web application
├── model.joblib                # Serialized trained LightGBM champion model (1.05 MB)
├── model_metrics.json          # Cached validation metrics & feature importances
├── sample_presets.json         # High and low propensity customer demo presets
├── sample_batch.csv            # 25-customer sample dataset for batch upload testing
├── train_model.py              # Reproducible training & artifact generation pipeline
├── requirements.txt            # Python dependencies for Streamlit Cloud
├── Report_on_Challenges_Faced.md # Comprehensive engineering report (PRCP-1003)
├── PRCP-1003_Customer_Transaction_Prediction.ipynb # Jupyter notebook
└── README.md                   # Project documentation
```

---

## 📜 License
This project is developed under the MIT License.
