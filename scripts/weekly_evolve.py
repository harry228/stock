#!/usr/bin/env python3
"""
TAIEX ML Self-Evolution and Weekly Hyperparameter Tournament (A/B Test Gate)
- Evaluates recent prediction accuracy from taiex_eval_history.csv.
- If drift is detected or as a scheduled weekly routine, generates challenger hyperparameter sets.
- Conducts a Walk-Forward Validation Tournament (Champion vs. Challengers) on historical data.
- Automatically promotes the winning hyperparameters if they meet the safety improvement threshold (1.5%+ MAE reduction).
- Syncs configurations and evolution reports to Google Drive.
"""

import os
import csv
import json
import random
import subprocess
import numpy as np
from datetime import datetime

# Import models
from train_taiex_multi_task_ensemble import (
    BaggingModel, BoostingModel, NeuralNetModel, StackingModel
)

DATA_DIR = os.path.expanduser("~/stock-analysis/data")
CSV_EVAL_PATH = os.path.join(DATA_DIR, "taiex_eval_history.csv")
HYPERPARAMS_PATH = os.path.join(DATA_DIR, "model_hyperparams.json")
TRAIN_DATA_PATH = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
if not os.path.exists(TRAIN_DATA_PATH):
    TRAIN_DATA_PATH = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")

DEFAULT_PARAMS = {
    "c_bag": {"n_estimators": 30, "max_depth": 5},
    "c_boost": {"n_estimators": 25, "lr": 0.08, "max_depth": 3},
    "c_nn": {"hidden": 32, "lr": 0.02, "epochs": 100},
    "r_bag": {"n_estimators": 30, "max_depth": 5},
    "r_boost": {"n_estimators": 25, "lr": 0.08, "max_depth": 3},
    "r_nn": {"hidden": 32, "lr": 0.01, "epochs": 120}
}

def load_current_params():
    if os.path.exists(HYPERPARAMS_PATH):
        try:
            with open(HYPERPARAMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_PARAMS

def evaluate_recent_performance(history_limit=10):
    """Calculates metrics on recent predictions from history file."""
    if not os.path.exists(CSV_EVAL_PATH):
        return {"status": "no_history", "msg": "歷史記錄檔不存在，需累積預測數據。"}
        
    completed_records = []
    with open(CSV_EVAL_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("actual_close") and r.get("dir_correct") != "":
                completed_records.append(r)
                
    if len(completed_records) < 5:
        return {"status": "low_data", "msg": f"已結算數據過少 ({len(completed_records)} 筆)，需累積至少 5 筆預測結算。"}
        
    recent = completed_records[-history_limit:]
    
    # Compute metrics
    direction_hits = [int(r["dir_correct"]) for r in recent]
    hit_rate = np.mean(direction_hits)
    
    pct_errors = [float(r["pct_error"]) for r in recent if r.get("pct_error")]
    avg_pct_err = np.mean(pct_errors) if pct_errors else 0.0
    
    return {
        "status": "ok",
        "sample_size": len(recent),
        "hit_rate": hit_rate,
        "avg_pct_error": avg_pct_err
    }

def generate_challengers(champion_params):
    """Generates 2 challenger parameter configurations by mutating champion parameters."""
    challengers = []
    for i in range(2):
        chal = {}
        # Mutate Bagging
        chal["c_bag"] = {
            "n_estimators": max(15, champion_params["c_bag"]["n_estimators"] + random.choice([-5, 5, 10])),
            "max_depth": max(3, champion_params["c_bag"]["max_depth"] + random.choice([-1, 1]))
        }
        # Mutate Boosting
        chal["c_boost"] = {
            "n_estimators": max(15, champion_params["c_boost"]["n_estimators"] + random.choice([-5, 5])),
            "lr": round(max(0.02, champion_params["c_boost"]["lr"] + random.choice([-0.02, 0.02])), 3),
            "max_depth": max(2, champion_params["c_boost"]["max_depth"] + random.choice([-1, 1]))
        }
        # Mutate NeuralNet
        chal["c_nn"] = {
            "hidden": random.choice([24, 32, 48]),
            "lr": round(max(0.005, champion_params["c_nn"]["lr"] + random.choice([-0.005, 0.005])), 3),
            "epochs": max(50, champion_params["c_nn"]["epochs"] + random.choice([-20, 20]))
        }
        
        # Sync regression to classification mutants with slight shifts
        chal["r_bag"] = chal["c_bag"].copy()
        chal["r_boost"] = chal["c_boost"].copy()
        chal["r_nn"] = {
            "hidden": chal["c_nn"]["hidden"],
            "lr": round(max(0.005, chal["c_nn"]["lr"] * 0.5), 3),
            "epochs": int(chal["c_nn"]["epochs"] * 1.2)
        }
        challengers.append(chal)
    return challengers

def evaluate_params_on_validation(params, X, labels, closes, horizons=[1, 3, 5], val_size=15):
    """Trains and tests a parameter config on local walk-forward validation set."""
    N = len(X)
    train_size = N - val_size
    
    val_maes = []
    val_accs = []
    
    for h in horizons:
        X_h, y_cls_h, y_reg_ratio_h = [], [], []
        for i in range(N - h):
            if X[i] is not None and labels[i + h] is not None:
                X_h.append(X[i])
                y_cls_h.append(labels[i + h])
                y_reg_ratio_h.append(closes[i + h] / closes[i])
                
        X_arr = np.array(X_h, dtype=np.float32)
        y_cls_arr = np.array(y_cls_h, dtype=np.int32)
        y_reg_ratio_arr = np.array(y_reg_ratio_h, dtype=np.float32)
        
        # Split into local train / val
        tr_sz = len(X_arr) - val_size
        X_tr, X_val = X_arr[:tr_sz], X_arr[tr_sz:]
        y_cls_tr, y_cls_val = y_cls_arr[:tr_sz], y_cls_arr[tr_sz:]
        y_reg_tr, y_reg_val = y_reg_ratio_arr[:tr_sz], y_reg_ratio_arr[tr_sz:]
        
        # Train Classification
        c_bag = BaggingModel(n_estimators=min(params["c_bag"]["n_estimators"], 15), max_depth=params["c_bag"]["max_depth"], is_regression=False)
        c_bag.fit(X_tr, y_cls_tr)
        c_boost = BoostingModel(n_estimators=min(params["c_boost"]["n_estimators"], 15), lr=params["c_boost"]["lr"], max_depth=params["c_boost"]["max_depth"], is_regression=False)
        c_boost.fit(X_tr, y_cls_tr)
        nn_epochs_c = min(params["c_nn"]["epochs"], 30)
        c_nn = NeuralNetModel(hidden=params["c_nn"]["hidden"], lr=params["c_nn"]["lr"], epochs=nn_epochs_c, is_regression=False)
        c_nn.fit(X_tr, y_cls_tr)
        c_stack = StackingModel(c_bag, c_boost, c_nn, is_regression=False)
        c_stack.fit(X_tr, y_cls_tr)
        
        # Train Regression
        r_bag = BaggingModel(n_estimators=min(params["r_bag"]["n_estimators"], 15), max_depth=params["r_bag"]["max_depth"], is_regression=True)
        r_bag.fit(X_tr, y_reg_tr)
        r_boost = BoostingModel(n_estimators=min(params["r_boost"]["n_estimators"], 15), lr=params["r_boost"]["lr"], max_depth=params["r_boost"]["max_depth"], is_regression=True)
        r_boost.fit(X_tr, y_reg_tr)
        nn_epochs_r = min(params["r_nn"]["epochs"], 30)
        r_nn = NeuralNetModel(hidden=params["r_nn"]["hidden"], lr=params["r_nn"]["lr"], epochs=nn_epochs_r, is_regression=True)
        r_nn.fit(X_tr, y_reg_tr)
        r_stack = StackingModel(r_bag, r_boost, r_nn, is_regression=True)
        r_stack.fit(X_tr, y_reg_tr)
        
        # Predictions
        pred_cls = c_stack.predict(X_val)
        pred_ratio = r_stack.predict(X_val)
        
        # Accuracy
        acc = np.mean(pred_cls == y_cls_val)
        val_accs.append(acc)
        
        # MAE of price ratios
        mae = np.mean(np.abs(pred_ratio - y_reg_val))
        val_maes.append(mae)
        
    return float(np.mean(val_maes)), float(np.mean(val_accs))

def main():
    print("=== TAIEX ML 自我進化超參數擂台對決 ===")
    
    # 1. Evaluate current performance
    perf = evaluate_recent_performance()
    print(f"\n【最新預測健康檢查】:")
    if perf["status"] == "ok":
        print(f"  - 評估樣本數: {perf['sample_size']} 天")
        print(f"  - 漲跌方向準確率 (Hit Rate): {perf['hit_rate']*100:.2f}%")
        print(f"  - 平均預測價格絕對誤差 (MAPE): {perf['avg_pct_error']:.2f}%")
        
        # Trigger condition (Hit Rate lower than 55% or routine evolution)
        if perf["hit_rate"] < 0.55:
            print("  - [診斷結果]: 效能出現漂移，觸發超參數強制進化！")
        else:
            print("  - [診斷結果]: 模型效能健康。執行例行性進化探索。")
    else:
        print(f"  - {perf['msg']} (套用例行性進化訓練)")

    # 2. Load historical data for Walk-Forward Validation
    print("\n載入歷史數據進行擂台模擬訓練...")
    rows = []
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    stationary_features = [
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal"
    ]
    available_features = [f for f in stationary_features if f in rows[0]]
    
    X_raw, labels_raw, closes_raw = [], [], []
    for r in rows:
        closes_raw.append(float(r["Close"]))
        labels_raw.append(int(r["Label"]) if r["Label"] != "" else None)
        vec = []
        valid = True
        for f in available_features:
            val = r.get(f, "")
            if val == "" or val is None: valid = False; break
            vec.append(float(val))
        X_raw.append(vec if valid else None)

    champion = load_current_params()
    challengers = generate_challengers(champion)
    
    all_competitors = [
        {"name": "Champion (現有模型)", "params": champion},
        {"name": "Challenger A (新星 1)", "params": challengers[0]},
        {"name": "Challenger B (新星 2)", "params": challengers[1]}
    ]
    
    print("\n>>> 開始擂台對決 (Walk-Forward Tournament) <<<")
    tournament_results = []
    for comp in all_competitors:
        print(f"  測試競賽者: {comp['name']}...")
        mae, acc = evaluate_params_on_validation(comp["params"], X_raw, labels_raw, closes_raw)
        comp["mae"] = mae
        comp["acc"] = acc
        tournament_results.append(comp)
        print(f"  結果 -> 平均驗證 MAE: {mae:.6f} | 方向準確率: {acc*100:.2f}%")
        
    # Sort by MAE (lower is better)
    tournament_results.sort(key=lambda x: x["mae"])
    best_model = tournament_results[0]
    
    # Compare Best with Champion
    champ_model = next(x for x in tournament_results if "Champion" in x["name"])
    
    improvement = (champ_model["mae"] - best_model["mae"]) / (champ_model["mae"] + 1e-9)
    print("\n" + "="*60)
    print("                   擂台對決最終成績單")
    print("="*60)
    for idx, r in enumerate(tournament_results):
        print(f" 第 {idx+1} 名: {r['name']:<20} | 驗證 MAE: {r['mae']:.6f} | 準確率: {r['acc']*100:.2f}%")
    print("-"*60)
    
    evolve_triggered = False
    report_lines = []
    report_lines.append("=== TAIEX ML 模型自我進化每週評估報告 ===")
    report_lines.append(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"最新實際預測表現 (近10日): 準確率 {perf.get('hit_rate', 0.0)*100:.2f}%, 平均絕對誤差 {perf.get('avg_pct_error', 0.0):.2f}%")
    report_lines.append("-" * 50)
    
    if best_model["name"] != champ_model["name"] and improvement >= 0.015:
        print(f"【進化成功】 {best_model['name']} 表現優於現有 Champion！")
        print(f"  - MAE 改善幅度: {improvement*100:.2f}% (突破 1.5% 安全晉級門檻)")
        print(f"  - 新方向準確率: {best_model['acc']*100:.2f}% (原 Champion: {champ_model['acc']*100:.2f}%)")
        print(">>> 執行超參數動態進化覆蓋！")
        
        with open(HYPERPARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(best_model["params"], f, ensure_ascii=False, indent=2)
            
        evolve_triggered = True
        report_lines.append(f"【模型自我進化】: 進化成功！【{best_model['name']}】晉級為新一代 Champion。")
        report_lines.append(f"  - MAE 降低幅度: {improvement*100:.2f}% (超越 1.5% 安全關卡)")
        report_lines.append(f"  - 舊驗證 MAE  : {champ_model['mae']:.6f} -> 新驗證 MAE  : {best_model['mae']:.6f}")
        report_lines.append(f"  - 舊驗證準確度: {champ_model['acc']*100:.2f}% -> 新驗證準確度: {best_model['acc']*100:.2f}%")
    else:
        print("【保留原模型】 未能找到能穩定顯著超越 (1.5% MAE) 目前 Champion 的新超參數組合。")
        print(">>> 放棄進化重置，以防止過度擬合與市場噪音污染。")
        report_lines.append("【模型自我進化】: 保留目前 Champion。")
        report_lines.append("  - 原因: Challengers 未能突破 1.5% 的安全改善率 (或表現不如原模型)，防禦過擬合鎖啟動。")
        
    report_lines.append("-" * 50)
    report_lines.append("【新世代模型超參數明細】:")
    curr_active = load_current_params()
    report_lines.append(json.dumps(curr_active, indent=2))
    
    # Save Weekly Report
    report_text = "\n".join(report_lines)
    report_path = os.path.expanduser("~/storage/downloads/taiex_self_evolution_weekly_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # Sync params and report to Google Drive
    print("\n備份超參數與自我進化報告至 Google Drive...")
    for f_path in [HYPERPARAMS_PATH, report_path]:
        if os.path.exists(f_path):
            try:
                subprocess.run(["rclone", "copy", f_path, "gdrive:stock-analysis/eval_logs/"], check=True)
                print(f"已同步至 GDrive: {os.path.basename(f_path)}")
            except Exception as e:
                print(f"GDrive 同步失敗 ({os.path.basename(f_path)}): {e}")
                
    # Send email report to user neolin909@gmail.com
    print("\n發送每週自我進化報告郵件...")
    subject = f"[EVOLUTION] TAIEX ML 自我進化每週擂台報告 - {datetime.now().strftime('%Y-%m-%d')}"
    email_cmd = [
        "python3", os.path.expanduser("~/send_email.py"),
        "--to", "neolin909@gmail.com",
        "--subject", subject,
        "--body", report_text
    ]
    try:
        subprocess.run(email_cmd, check=True)
        print("進化報告郵件發送成功！")
    except Exception as e:
        print(f"進化報告郵件發送失敗: {e}")

    # 執行模型備份 (Backup Models)
    print("\n執行模型檔案備份至 GDrive...")
    backup_cmd = ["python3", os.path.expanduser("~/stock-analysis/scripts/backup_models.py")]
    try:
        subprocess.run(backup_cmd, check=True)
        print("模型備份作業完成！")
    except Exception as e:
        print(f"模型備份失敗: {e}")
                
    print("\n每週自我進化對決圓滿完成！")

if __name__ == "__main__":
    main()
