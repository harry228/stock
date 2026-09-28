#!/usr/bin/env python3
"""
TAIEX Ensemble Prediction Evaluation & Residual Tracker
- Records daily ML multi-horizon predictions (1d, 2d, 3d, 4d, 5d).
- Automatically backfills actual prices and labels from latest market data.
- Computes price residuals, absolute percentage errors, and direction accuracy.
- Syncs prediction records and evaluation logs to Google Drive (gdrive:stock-analysis/eval_logs/).
"""

import os
import csv
import json
import subprocess
from datetime import datetime

DATA_DIR = os.path.expanduser("~/stock-analysis/data")
JSON_TRACKER_PATH = os.path.join(DATA_DIR, "taiex_prediction_tracker.json")
CSV_EVAL_PATH = os.path.join(DATA_DIR, "taiex_eval_history.csv")
TRAIN_DATA_PATH = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
if not os.path.exists(TRAIN_DATA_PATH):
    TRAIN_DATA_PATH = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")

def load_tracker():
    """Loads prediction tracker records from JSON file."""
    if os.path.exists(JSON_TRACKER_PATH):
        try:
            with open(JSON_TRACKER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[eval_tracker] 讀取 tracker 失敗: {e}，建立新紀錄檔。")
    return {"records": []}

def save_tracker(tracker_data):
    """Saves prediction tracker records to JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JSON_TRACKER_PATH, "w", encoding="utf-8") as f:
        json.dump(tracker_data, f, ensure_ascii=False, indent=2)

def record_prediction(base_date, base_close, horizons_data):
    """
    Records a new daily multi-horizon prediction.
    horizons_data format:
    {
      1: {"target_date": "YYYY-MM-DD", "pred_price": 22000.0, "pred_label": 1},
      2: {"target_date": "YYYY-MM-DD", "pred_price": 22050.0, "pred_label": 1},
      ...
    }
    """
    tracker = load_tracker()
    
    # Check if prediction for this base_date already exists
    existing_idx = None
    for idx, rec in enumerate(tracker["records"]):
        if rec["base_date"] == base_date:
            existing_idx = idx
            break
            
    entry = {
        "base_date": base_date,
        "base_close": float(base_close),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizons": {}
    }
    
    for h, hdata in horizons_data.items():
        entry["horizons"][str(h)] = {
            "target_date": hdata["target_date"],
            "pred_price": round(float(hdata["pred_price"]), 2),
            "pred_label": int(hdata["pred_label"]),
            "actual_close": hdata.get("actual_close", None),
            "actual_label": hdata.get("actual_label", None)
        }
        
    if existing_idx is not None:
        tracker["records"][existing_idx] = entry
        print(f"[eval_tracker] 更新基準日 {base_date} 的預測紀錄。")
    else:
        tracker["records"].append(entry)
        print(f"[eval_tracker] 新增基準日 {base_date} 的預測紀錄。")
        
    save_tracker(tracker)
    update_actuals_and_residuals()

def load_market_data():
    """Loads actual historical prices and labels into a lookup map."""
    market_map = {}
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[eval_tracker] 警告: 找不到歷史資料檔 {TRAIN_DATA_PATH}")
        return market_map
        
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dt = r.get("Date", "").strip()
            close_str = r.get("Close", "").strip()
            label_str = r.get("Label", "").strip()
            if dt and close_str:
                market_map[dt] = {
                    "close": float(close_str),
                    "label": int(label_str) if label_str != "" else None
                }
    return market_map

def update_actuals_and_residuals():
    """
    Backfills actual market prices/labels and exports CSV evaluation report.
    """
    tracker = load_tracker()
    market_map = load_market_data()
    
    if not tracker["records"]:
        print("[eval_tracker] 無任何預測紀錄可供回填。")
        return
        
    updated_count = 0
    csv_rows = []
    
    for rec in tracker["records"]:
        b_date = rec["base_date"]
        b_close = rec["base_close"]
        
        for h_str, hinfo in rec["horizons"].items():
            t_date = hinfo["target_date"]
            
            # Try backfilling if actual values missing
            if (hinfo["actual_close"] is None or hinfo["actual_label"] is None) and t_date in market_map:
                actual_info = market_map[t_date]
                hinfo["actual_close"] = actual_info["close"]
                hinfo["actual_label"] = actual_info["label"]
                updated_count += 1
                
            pred_p = hinfo["pred_price"]
            pred_l = hinfo["pred_label"]
            act_p = hinfo["actual_close"]
            act_l = hinfo["actual_label"]
            
            # Compute residual metrics if actuals are available
            price_err = round(pred_p - act_p, 2) if act_p is not None else None
            abs_err = round(abs(price_err), 2) if price_err is not None else None
            pct_err = round((abs_err / act_p) * 100.0, 2) if (abs_err is not None and act_p) else None
            
            dir_correct = None
            if act_l is not None:
                dir_correct = 1 if (pred_l == act_l) else 0
                
            csv_rows.append({
                "base_date": b_date,
                "base_close": b_close,
                "horizon_days": int(h_str),
                "target_date": t_date,
                "pred_price": pred_p,
                "actual_close": act_p if act_p is not None else "",
                "price_error": price_err if price_err is not None else "",
                "abs_error": abs_err if abs_err is not None else "",
                "pct_error": pct_err if pct_err is not None else "",
                "pred_label": pred_l,
                "actual_label": act_l if act_l is not None else "",
                "dir_correct": dir_correct if dir_correct is not None else ""
            })
            
    if updated_count > 0:
        save_tracker(tracker)
        print(f"[eval_tracker] 已成功回填 {updated_count} 筆實際交易日數據。")
        
    # Write to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = [
        "base_date", "base_close", "horizon_days", "target_date",
        "pred_price", "actual_close", "price_error", "abs_error",
        "pct_error", "pred_label", "actual_label", "dir_correct"
    ]
    
    with open(CSV_EVAL_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"[eval_tracker] 評估歷史記錄已導出: {CSV_EVAL_PATH} (共 {len(csv_rows)} 筆天數比對)")

def sync_to_cloud():
    """Syncs tracker records and CSV history to Google Drive."""
    print("[eval_tracker] 開始備份預測日誌與評估記錄至 Google Drive...")
    gdrive_target = "gdrive:stock-analysis/eval_logs/"
    
    for file_path in [JSON_TRACKER_PATH, CSV_EVAL_PATH]:
        if os.path.exists(file_path):
            try:
                subprocess.run(["rclone", "copy", file_path, gdrive_target], check=True)
                print(f"[eval_tracker] 已備份至雲端: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[eval_tracker] 雲端備份失敗 ({os.path.basename(file_path)}): {e}")

if __name__ == "__main__":
    print("=== TAIEX 評估追蹤模組 ===")
    update_actuals_and_residuals()
    sync_to_cloud()
