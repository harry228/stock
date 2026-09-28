#!/usr/bin/env python3
"""
TAIEX ML Model Daily Inference Script
To be executed on Tablet (Termux) or Desktop.
Loads trained RandomForest classifier to predict current market regime state.
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def main():
    model_path = os.path.expanduser("~/stock-analysis/models/taiex_rf_model.joblib")
    data_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    
    if not os.path.exists(model_path):
        # Fallback to local
        model_path = "taiex_rf_model.joblib"
        if not os.path.exists(model_path):
            print(f"Error: Trained model file not found at {model_path}.")
            print("Please train the model on your Desktop first and copy 'taiex_rf_model.joblib' to ~/stock-analysis/models/")
            return
            
    if not os.path.exists(data_path):
        print(f"Error: Feature CSV file not found at {data_path}.")
        print("Please run 'python3 ~/stock-analysis/scripts/generate_ml_labels.py' first to generate latest features.")
        return

    # 1. Load Model and Metadata
    print(f"Loading Model: {model_path}")
    model_meta = joblib.load(model_path)
    model = model_meta["model"]
    feature_cols = model_meta["features"]
    trained_date = model_meta.get("trained_date", "Unknown")
    trained_acc = model_meta.get("accuracy", 0.0)
    
    print(f"Model trained on: {trained_date} (Accuracy: {trained_acc:.2%})")

    # 2. Load and parse the latest market data
    df = pd.read_csv(data_path)
    if len(df) == 0:
        print("Error: Feature CSV is empty.")
        return
        
    # Get the last row (the latest trading day)
    latest_row = df.iloc[-1]
    latest_date = latest_row["Date"]
    
    print(f"Latest Available Market Data Date: {latest_date}")
    
    # Check data freshness (if it's too old, warn the user)
    # Note: weekends/holidays are expected, but warning if > 5 days old
    try:
        data_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        days_diff = (datetime.now() - data_dt).days
        if days_diff > 5:
            print(f"\n[WARNING] Market data is {days_diff} days old! Please update cache first:")
            print("  python3 ~/stock-analysis/scripts/taiex_rsi_divergence.py --update --restore-weight")
            print("  python3 ~/stock-analysis/scripts/generate_ml_labels.py")
            print("-" * 50)
    except Exception as e:
        pass

    # Extract features for prediction
    features_df = pd.DataFrame([latest_row[feature_cols]])
    
    # Check for NaNs in the inference row
    if features_df.isnull().any().any():
        nan_cols = features_df.columns[features_df.isnull().any()].tolist()
        print(f"\nError: The latest data row contains missing features: {nan_cols}")
        print("Please ensure you have enough historical cache to compute indicators (at least 120 days).")
        return

    # 3. Perform Inference
    prediction = model.predict(features_df)[0]
    probs = model.predict_proba(features_df)[0] # Class order matches model.classes_
    
    # Map predictions
    label_map = {1: "上漲 (Up)", 0: "盤整 (Consolidation)", -1: "下跌 (Down)"}
    classes = model.classes_
    prob_dict = {label_map[c]: probs[i] for i, c in enumerate(classes)}
    
    # Output inference results
    print("\n" + "="*50)
    print("         TAIEX ML REGIME INFERENCE REPORT")
    print("="*50)
    print(f"  分析標的  : 台股加權指數 (^TWII)")
    print(f"  資料日期  : {latest_date}")
    print(f"  預測狀態  : 【 {label_map[prediction]} 】")
    print("-" * 50)
    print("  各狀態預測機率分佈 (Confidence Levels):")
    for state, prob in prob_dict.items():
        bar = "█" * int(prob * 20)
        print(f"    - {state:20s}: {prob:6.2%}  {bar}")
    print("="*50)
    
    # Suggest actions based on prediction
    print("\n  [操作建議]")
    if prediction == 1:
        print("    >> 多頭趨勢確認。建議偏多操作，維持高持股水位。")
    elif prediction == -1:
        print("    >> 空頭趨勢確認。建議避險、降低部位或保留高現金水位。")
    else:
        print("    >> 盤整盤勢。建議採取區間操作（低吸高拋），或縮減部位曝險比例 (例如 0.3x-0.5x)。")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
