#!/usr/bin/env python3
"""
TAIEX ML Model Training Script
To be executed on Desktop Agent.
Trains a RandomForest Classifier on TAIEX dual-threshold labeling dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def main():
    data_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
    if not os.path.exists(data_path):
        # Fallback to local working directory
        data_path = "taiex_ml_training_data.csv"
        if not os.path.exists(data_path):
            print("Error: Training data not found. Please run generate_ml_labels.py first.")
            return

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Define Features & Target
    # Exclude non-stationary absolute price and volume columns to prevent overfitting to specific eras
    non_stationary_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_3", "SMA_5", "SMA_10", "SMA_20", "SMA_60", "SMA_120",
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]
    drop_cols = ["Date", "State", "Label"] + non_stationary_cols + [col for col in df.columns if "Future_Return" in col]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    print("\nSelected Features:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")
        
    # Drop rows with NaN values in features or target
    # (Indicators like SMAs, KD, RSI need initial days to warm up)
    df_clean = df.dropna(subset=feature_cols + ["Label"]).copy()
    print(f"\nOriginal data points: {len(df)}")
    print(f"Cleaned data points (after removing warm-up NaNs): {len(df_clean)}")
    
    X = df_clean[feature_cols]
    y = df_clean["Label"].astype(int)
    
    # 2. Chronological Train-Test Split (Prevent temporal leakage)
    # Train: 80%, Test: 20%
    split_idx = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples ({df_clean['Date'].iloc[0]} ~ {df_clean['Date'].iloc[split_idx-1]})")
    print(f"Testing set:  {len(X_test)} samples ({df_clean['Date'].iloc[split_idx]} ~ {df_clean['Date'].iloc[-1]})")
    
    # 3. Model Setup & Training
    # We use class_weight='balanced' to handle the natural class imbalance of the stock market
    print("\nTraining RandomForest Classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down (-1)", "Consolidation (0)", "Up (1)"]))
    
    # Feature Importance Analysis
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for f in range(X.shape[1]):
        print(f"  {f+1:2d}. {X.columns[indices[f]]:20s} : {importances[indices[f]]:.4f}")
        
    # 5. Export Model & Metadata
    model_dir = os.path.expanduser("~/stock-analysis/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "taiex_rf_model.joblib")
    
    model_meta = {
        "model": model,
        "features": feature_cols,
        "trained_date": datetime.now().strftime("%Y-%m-%d"),
        "accuracy": acc
    }
    
    joblib.dump(model_meta, model_path)
    print(f"\n[Success] Model successfully exported to: {model_path}")

if __name__ == "__main__":
    from datetime import datetime
    main()
