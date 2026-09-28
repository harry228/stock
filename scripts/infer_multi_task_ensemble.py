#!/usr/bin/env python3
"""
Inference script for TAIEX Multi-Task Ensemble Pipeline.
Trains models on all available data, predicts the future 1-day, 3-day, and 5-day 
market direction and index close prices based on the latest available market data, 
and sends an email report.
"""

import os
import csv
import json
import time
import subprocess
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Attempt to set Chinese font for Android/Termux
font_path = "/system/fonts/MiSansTCVF.ttf"
if not os.path.exists(font_path):
    font_path = "/system/fonts/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    try:
        fe = fm.FontEntry(fname=font_path, name='CustomFont')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.family'] = fe.name
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# Load classes from the training script
from train_taiex_multi_task_ensemble import (
    BaggingModel, BoostingModel, NeuralNetModel, StackingModel
)
import eval_tracker

def get_next_trading_days(start_date_str, count=5):
    """
    Simulates subsequent trading dates (excluding weekends).
    """
    curr = datetime.strptime(start_date_str, '%Y-%m-%d')
    days = []
    while len(days) < count:
        curr += timedelta(days=1)
        if curr.weekday() < 5: # Monday to Friday
            days.append(curr.strftime('%Y-%m-%d'))
    return days

def main():
    print("更新台股歷史數據與生成最新 ML 標籤...")
    subprocess.run("python3 ~/stock-analysis/scripts/taiex_rsi_divergence.py --update --restore-weight", shell=True, capture_output=True)
    subprocess.run("python3 ~/stock-analysis/scripts/generate_ml_labels.py", shell=True, capture_output=True)

    print("Loading data...")
    csv_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)

    N = len(rows)
    latest_row = rows[-1]
    # 取得當前日期與時間
    today = datetime.now()
    
    # 判斷是否處於台股收盤時間後
    market_closed = today.hour >= 14 or (today.hour == 13 and today.minute >= 30)
    expected_date = today.date() if market_closed else (today - timedelta(days=1)).date()
    
    # 讀取 CSV 資料日期與收盤價
    latest_row = rows[-1]
    latest_date_str = latest_row["Date"]
    latest_close = float(latest_row["Close"])
    
    latest_dt = datetime.strptime(latest_date_str, "%Y-%m-%d")
    
    # 判斷是否處於台股收盤時間後
    market_closed = today.hour >= 14 or (today.hour == 13 and today.minute >= 30)
    
    # 處理週末與交易日對齊
    if today.weekday() == 6: # 週日
        expected_date = (today - timedelta(days=2)).date()
    elif today.weekday() == 5: # 週六
        expected_date = (today - timedelta(days=1)).date()
    elif today.weekday() == 0 and not market_closed: # 週一開盤前
        expected_date = (today - timedelta(days=3)).date()
    elif not market_closed: # 其他工作日收盤前，預期前一交易日
        expected_date = (today - timedelta(days=1)).date()
    else: # 工作日收盤後，預期今日
        expected_date = today.date()

    delta = (expected_date - latest_dt.date()).days
    print(f"最新數據基準日: {latest_date_str}, 收盤價: {latest_close}, 預期交易日: {expected_date}, 差距: {delta}")
    
    if delta > 1:
        print(f"【警報】資料未同步至最新收盤日 (預期: {expected_date})，中止作業。")
        import sys
        sys.exit(1)
    elif delta == 1:
        print(f"【提醒】數據延遲一日，嘗試推演...")

    stationary_features = [
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal"
    ]
    available_features = [f for f in stationary_features if f in rows[0]]

    X_raw, labels_raw, closes_raw, dates = [], [], [], []
    for r in rows:
        dates.append(r["Date"])
        closes_raw.append(float(r["Close"]))
        labels_raw.append(int(r["Label"]) if r["Label"] != "" else None)
        vec = []
        valid = True
        for f in available_features:
            val = r.get(f, "")
            if val == "" or val is None: valid = False; break
            vec.append(float(val))
        X_raw.append(vec if valid else None)

    closes_arr = np.array(closes_raw)
    
    # Extract feature vector for the latest day (for inference)
    latest_x_feat = X_raw[-1]
    if latest_x_feat is None:
        # Find the last valid row
        for idx in range(N - 1, -1, -1):
            if X_raw[idx] is not None:
                latest_x_feat = X_raw[idx]
                latest_date_str = dates[idx]
                latest_close = closes_raw[idx]
                break

    X_infer = np.array([latest_x_feat], dtype=np.float32)

    predictions = {}
    horizons = [1, 2, 3, 4, 5, 10, 15, 20]
    label_desc = {1: "上漲 (Up)", 0: "盤整 (Consolidation)", -1: "下跌 (Down)"}
    
    # Simulate trading days up to 20 days
    next_days = get_next_trading_days(latest_date_str, 20)
    horizon_dates = {h: next_days[h - 1] for h in horizons}

    print("Training models on entire history and running multi-task inference...")
    
    # Load dynamic hyperparameters
    hyperparams_path = os.path.join(eval_tracker.DATA_DIR, "model_hyperparams.json")
    default_params = {
        "c_bag": {"n_estimators": 30, "max_depth": 5},
        "c_boost": {"n_estimators": 25, "lr": 0.08, "max_depth": 3},
        "c_nn": {"hidden": 32, "lr": 0.02, "epochs": 100},
        "r_bag": {"n_estimators": 30, "max_depth": 5},
        "r_boost": {"n_estimators": 25, "lr": 0.08, "max_depth": 3},
        "r_nn": {"hidden": 32, "lr": 0.01, "epochs": 120}
    }
    
    if os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, "r", encoding="utf-8") as f:
                hparams = json.load(f)
                print("[infer_ensemble] 成功載入動態進化超參數配置。")
        except Exception as e:
            print(f"[infer_ensemble] 讀取超參數檔案失敗: {e}，套用預設值。")
            hparams = default_params
    else:
        hparams = default_params

    for h in horizons:
        X_h, y_cls_h, y_reg_ratio_h = [], [], []
        for i in range(N - h):
            if X_raw[i] is not None and labels_raw[i + h] is not None:
                X_h.append(X_raw[i])
                y_cls_h.append(labels_raw[i + h])
                y_reg_ratio_h.append(closes_arr[i + h] / closes_arr[i])

        X_arr = np.array(X_h, dtype=np.float32)
        y_cls_arr = np.array(y_cls_h, dtype=np.int32)
        y_reg_ratio_arr = np.array(y_reg_ratio_h, dtype=np.float32)

        # 1. Classification models
        c_bag = BaggingModel(
            n_estimators=hparams["c_bag"]["n_estimators"], 
            max_depth=hparams["c_bag"]["max_depth"], 
            is_regression=False
        )
        c_bag.fit(X_arr, y_cls_arr)

        c_boost = BoostingModel(
            n_estimators=hparams["c_boost"]["n_estimators"], 
            lr=hparams["c_boost"]["lr"], 
            max_depth=hparams["c_boost"]["max_depth"], 
            is_regression=False
        )
        c_boost.fit(X_arr, y_cls_arr)

        c_nn = NeuralNetModel(
            hidden=hparams["c_nn"]["hidden"], 
            lr=hparams["c_nn"]["lr"], 
            epochs=hparams["c_nn"]["epochs"], 
            is_regression=False
        )
        c_nn.fit(X_arr, y_cls_arr)

        c_stack = StackingModel(c_bag, c_boost, c_nn, is_regression=False)
        c_stack.fit(X_arr, y_cls_arr)

        # 2. Regression models
        r_bag = BaggingModel(
            n_estimators=hparams["r_bag"]["n_estimators"], 
            max_depth=hparams["r_bag"]["max_depth"], 
            is_regression=True
        )
        r_bag.fit(X_arr, y_reg_ratio_arr)

        r_boost = BoostingModel(
            n_estimators=hparams["r_boost"]["n_estimators"], 
            lr=hparams["r_boost"]["lr"], 
            max_depth=hparams["r_boost"]["max_depth"], 
            is_regression=True
        )
        r_boost.fit(X_arr, y_reg_ratio_arr)

        r_nn = NeuralNetModel(
            hidden=hparams["r_nn"]["hidden"], 
            lr=hparams["r_nn"]["lr"], 
            epochs=hparams["r_nn"]["epochs"], 
            is_regression=True
        )
        r_nn.fit(X_arr, y_reg_ratio_arr)

        r_stack = StackingModel(r_bag, r_boost, r_nn, is_regression=True)
        r_stack.fit(X_arr, y_reg_ratio_arr)

        # Inference for horizon
        pred_cls_stack = int(c_stack.predict(X_infer)[0])
        prob_cls_stack = c_stack.predict_proba(X_infer)[0]

        pred_ratio_stack = float(r_stack.predict(X_infer)[0])
        pred_price_stack = latest_close * pred_ratio_stack

        predictions[f"{h}d"] = {
            "target_date": horizon_dates[h],
            "class_prediction": label_desc[pred_cls_stack],
            "probs": {
                "Down (-1)": round(float(prob_cls_stack[0]), 4),
                "Consolidation (0)": round(float(prob_cls_stack[1]), 4),
                "Up (1)": round(float(prob_cls_stack[2]), 4)
            },
            "predicted_price": round(pred_price_stack, 2),
            "predicted_change_pct": round((pred_ratio_stack - 1.0) * 100.0, 2),
            "raw_label": pred_cls_stack
        }

    # Record predictions into evaluation tracker & backfill actuals
    try:
        tracker_horizons = {}
        for h in horizons:
            p = predictions[f"{h}d"]
            tracker_horizons[h] = {
                "target_date": p["target_date"],
                "pred_price": p["predicted_price"],
                "pred_label": p["raw_label"]
            }
        eval_tracker.record_prediction(latest_date_str, latest_close, tracker_horizons)
    except Exception as e:
        print(f"【警告】記錄每日預測與評估日誌失敗: {e}")

    # Generate Report Text
    report_body = []
    report_body.append("="*75)
    report_body.append("     TAIEX MULTI-TASK ENSEMBLE FUTURE MARKET OUTLOOK")
    report_body.append("     基於純 NumPy (Bagging, Boosting, Stacking) 多任務預測系統")
    report_body.append("="*75)
    report_body.append(f"【最新數據基準日】: {latest_date_str}")
    report_body.append(f"【大盤最新收盤價】: {latest_close:,.2f} 點")
    report_body.append("【預估期間】: 未來 1~5 日短線與 10/15/20 日中線多任務預估折線趨勢")
    report_body.append("-"*75)

    # 擷取近 3 個月 (90 天) 歷史收盤價數據
    cutoff_dt = latest_dt - timedelta(days=90)
    hist_dates = []
    hist_closes = []
    for r in rows:
        r_dt = datetime.strptime(r["Date"], "%Y-%m-%d")
        if r_dt >= cutoff_dt:
            hist_dates.append(r["Date"])
            hist_closes.append(float(r["Close"]))

    # 短線 1~5 日預估資料點
    short_future_dates = [latest_date_str] + [predictions[f"{h}d"]["target_date"] for h in [1, 2, 3, 4, 5]]
    short_future_prices = [latest_close] + [predictions[f"{h}d"]["predicted_price"] for h in [1, 2, 3, 4, 5]]

    # 中長線 10, 15, 20 日預估資料點
    ext_future_dates = [predictions[f"{h}d"]["target_date"] for h in [10, 15, 20]]
    ext_future_prices = [predictions[f"{h}d"]["predicted_price"] for h in [10, 15, 20]]

    for h in horizons:
        p = predictions[f"{h}d"]
        term_tag = "短線" if h <= 5 else "中長線"
        report_body.append(f"▶【預測未來 {h:2d} 日後 [{term_tag}] ({p['target_date']}) 行情展望】:")
        report_body.append(f"  - 走勢方向分類預測: 【 {p['class_prediction']} 】")
        report_body.append(f"    [信心度機率]: 下跌 {p['probs']['Down (-1)']*100:.2f}% | 盤整 {p['probs']['Consolidation (0)']*100:.2f}% | 上漲 {p['probs']['Up (1)']*100:.2f}%")
        sign = "+" if p['predicted_change_pct'] >= 0 else ""
        report_body.append(f"  - 預估指數收盤價  : {p['predicted_price']:,.2f} 點 (預估變動: {sign}{p['predicted_change_pct']}%)")
        report_body.append("-"*75)

    # Generate Line Chart combining 3-month historical close prices, 5-day short forecast, and 10/15/20-day medium forecast
    chart_path = os.path.expanduser("~/storage/downloads/taiex_forecast_chart.png")
    try:
        plt.figure(figsize=(12, 6.5))
        
        # 1. 近 3 個月歷史收盤價 (黑色)
        plt.plot(hist_dates, hist_closes, linestyle='-', color='black', linewidth=1.8, label='近 3 個月實際收盤價')
        
        # 2. 未來 5 日預估價 (綠色實點連線)
        plt.plot(short_future_dates, short_future_prices, marker='o', linestyle='--', color='green', linewidth=2.2, label='未來 1~5 日短線預估價')
        
        # 3. 未來 10/15/20 日預估價 (橘色方塊點)
        plt.plot(ext_future_dates, ext_future_prices, marker='s', linestyle=':', color='darkorange', linewidth=2.0, label='未來 10/15/20 日中線預估點')
        
        # 為未來 1~5 日與基準日數據點標註點數
        for i, txt in enumerate(short_future_prices):
            color = 'green' if i > 0 else 'black'
            plt.annotate(
                f"{txt:,.0f}", 
                (short_future_dates[i], short_future_prices[i]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8.5, 
                color=color,
                weight='bold' if i > 0 else 'normal'
            )
            
        # 為 10/15/20 日數據點標註點數
        for i, txt in enumerate(ext_future_prices):
            plt.annotate(
                f"{txt:,.0f}", 
                (ext_future_dates[i], ext_future_prices[i]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8.5, 
                color='darkorange',
                weight='bold'
            )
        
        plt.title(f"台股加權指數近 3 個月收盤價與未來 1~20 日多天期預估趨勢圖 ({latest_date_str} 基準)")
        plt.xlabel("日期")
        plt.ylabel("指數點數")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 選擇精簡刻度標籤以防重疊
        all_future_dates = short_future_dates[1:] + ext_future_dates
        step = max(1, len(hist_dates) // 8)
        selected_ticks = hist_dates[::step]
        if hist_dates[-1] not in selected_ticks:
            selected_ticks.append(hist_dates[-1])
        for fd in all_future_dates:
            if fd not in selected_ticks:
                selected_ticks.append(fd)
                
        plt.xticks(selected_ticks, rotation=35, fontsize=8)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"預測折線圖已生成: {chart_path}")
    except Exception as e:
        print(f"折線圖生成失敗: {e}")
        chart_path = None

    report_body.append("\n【免責聲明】: 本報告僅供學術研究與 AI 機器學習模型效能測試之用，不構成任何實際投資建議。投資有風險，入市需謹慎。")

    email_content = "\n".join(report_body)
    print("\nGenerated Future Prediction Report:\n")
    print(email_content)

    # Save to dynamic report
    report_txt_path = os.path.expanduser("~/storage/downloads/taiex_future_outlook.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(email_content)

    # Sync to Google Drive
    print("\nSyncing report and chart to Google Drive...")
    for f_path in [report_txt_path, chart_path]:
        if f_path and os.path.exists(f_path):
            try:
                subprocess.run(["rclone", "copy", f_path, "gdrive:stock-analysis/"], check=True)
                print(f"已同步至 Google Drive: {f_path}")
            except Exception as e:
                print(f"同步至 Google Drive 失敗 ({f_path}): {e}")

    try:
        eval_tracker.sync_to_cloud()
    except Exception as e:
        print(f"評估日誌雲端備份失敗: {e}")

    print("\n訓練與檔案產出完成，寄信邏輯已由 send_ensemble_report.py 接管。")

if __name__ == "__main__":
    main()
