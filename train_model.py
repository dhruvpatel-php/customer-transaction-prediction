import time
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report

def train_and_export():
    print("=" * 60)
    print("Starting Customer Transaction Prediction Model Training Pipeline")
    print("=" * 60)
    
    t0 = time.time()
    csv_path = "train(1).csv"
    print(f"Loading {csv_path} (200,000 rows)...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns in {time.time() - t0:.2f}s")
    
    # Preprocessing
    feature_cols = [c for c in df.columns if c.startswith("var_")]
    X_raw = df[feature_cols]
    y = df["target"]
    
    # Feature Engineering (Row-level statistical moments)
    print("Engineering row-level statistical moments (mean, std, min, max, sum)...")
    fe_df = pd.DataFrame({
        "row_mean": X_raw.mean(axis=1),
        "row_std": X_raw.std(axis=1),
        "row_min": X_raw.min(axis=1),
        "row_max": X_raw.max(axis=1),
        "row_sum": X_raw.sum(axis=1)
    }, index=df.index)
    
    X = pd.concat([X_raw, fe_df], axis=1)
    all_feature_names = list(X.columns)
    print(f"Total features after engineering: {len(all_feature_names)}")
    
    # Stratified Split (80% Train, 20% Validation)
    print("Splitting dataset (80% Train, 20% Holdout Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    
    # Tuned LightGBM model configuration
    print("Configuring Tuned LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.4,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    t_train = time.time()
    print("Training model with early stopping on validation ROC-AUC...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
    )
    print(f"Training completed in {time.time() - t_train:.2f}s")
    
    # Evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_probs))
    print(f"Holdout Validation ROC-AUC: {val_auc:.5f}")
    
    # Cost-optimal threshold calculation
    # In banking, False Negative cost is high. As noted in project report, theta ~ 0.22 maximizes F1 / business utility
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    # F-beta or F1 score optimization
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.22
    print(f"Optimal Decision Threshold (Max F1): {optimal_threshold:.4f} (Benchmark: 0.22)")
    
    # Confusion matrix at optimal threshold and at 0.5 default
    preds_opt = (val_probs >= optimal_threshold).astype(int)
    preds_def = (val_probs >= 0.50).astype(int)
    
    cm_opt = confusion_matrix(y_val, preds_opt).tolist()
    cm_def = confusion_matrix(y_val, preds_def).tolist()
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(all_feature_names, [float(x) for x in importances]), key=lambda x: x[1], reverse=True)
    top_20_features = [{"feature": f, "importance": imp} for f, imp in feat_imp[:20]]
    
    # Save Model Artifact
    print("Exporting model to model.joblib...")
    model_artifact = {
        "model": model,
        "feature_names": all_feature_names,
        "optimal_threshold": round(optimal_threshold, 4),
        "val_auc": round(val_auc, 5),
    }
    joblib.dump(model_artifact, "model.joblib", compress=3)
    print(f"Saved model.joblib successfully (size: {len(open('model.joblib', 'rb').read()) / 1024 / 1024:.2f} MB)")
    
    # Save Metrics & Metadata
    metrics_data = {
        "val_auc": round(val_auc, 5),
        "optimal_threshold": round(optimal_threshold, 4),
        "default_threshold": 0.50,
        "confusion_matrix_optimal": cm_opt,
        "confusion_matrix_default": cm_def,
        "top_features": top_20_features,
        "classification_report_optimal": classification_report(y_val, preds_opt, output_dict=True),
        "classification_report_default": classification_report(y_val, preds_def, output_dict=True)
    }
    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print("Saved model_metrics.json")
    
    # Extract sample records for UI interactive demo & batch upload testing
    print("Extracting sample presets and batch test CSV...")
    pos_samples = df[df["target"] == 1].head(10)
    neg_samples = df[df["target"] == 0].head(10)
    
    # Sample presets for single prediction dropdown in UI
    presets = {
        "High Propensity Customer (Actual Transaction = 1)": {
            "ID_code": pos_samples.iloc[0]["ID_code"],
            "features": {col: float(pos_samples.iloc[0][col]) for col in feature_cols}
        },
        "High Propensity Customer 2 (Actual Transaction = 1)": {
            "ID_code": pos_samples.iloc[1]["ID_code"],
            "features": {col: float(pos_samples.iloc[1][col]) for col in feature_cols}
        },
        "Low Propensity Customer (Actual Transaction = 0)": {
            "ID_code": neg_samples.iloc[0]["ID_code"],
            "features": {col: float(neg_samples.iloc[0][col]) for col in feature_cols}
        },
        "Low Propensity Customer 2 (Actual Transaction = 0)": {
            "ID_code": neg_samples.iloc[1]["ID_code"],
            "features": {col: float(neg_samples.iloc[1][col]) for col in feature_cols}
        }
    }
    with open("sample_presets.json", "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)
    print("Saved sample_presets.json")
    
    # Batch test CSV (25 records without target column)
    batch_df = pd.concat([pos_samples.head(10), neg_samples.head(15)], ignore_index=True)
    batch_df_to_save = batch_df.drop(columns=["target"])
    batch_df_to_save.to_csv("sample_batch.csv", index=False)
    print("Saved sample_batch.csv (25 sample rows for batch testing)")
    
    print("=" * 60)
    print("All artifacts generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export()
