#!/usr/bin/env python3
"""
TAIEX Machine Learning Daily Report Job (Batched & Resilient Version)
- Implements batched execution with retry logic and rate-limit mitigation (429 handling) for Termux/Android.
- Kept alongside the original for testing and comparison.
"""

import os
import sys
import json
import csv
import time
import subprocess
from datetime import datetime

# Add project directory to sys.path
project_dir = os.path.expanduser("~/stock-analysis")
sys.path.append(project_dir)

try:
    from send_email import send_email
except ImportError:
    sys.path.append(os.path.expanduser("~"))
    from send_email import send_email

def run_step_with_retry(cmd, desc, max_retries=3, delay=10):
    """Executes a step with exponential backoff retry for stability against 429 / network errors."""
    for attempt in range(1, max_retries + 1):
        print(f"\n[步驟] {desc} (嘗試 {attempt}/{max_retries})...")
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                print(f"成功 ({desc})")
                return res.stdout
            else:
                print(f"警告 ({desc}) 失敗，返回碼 {res.returncode}: {res.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"警告 ({desc}) 執行超時 (Timeout)")
        
        if attempt < max_retries:
            print(f"等待 {delay} 秒後進行第 {attempt + 1} 次重試...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            
    raise RuntimeError(f"{desc} 在重試 {max_retries} 次後仍然失敗。")

def main():
    print(f"=== 開始執行 [Batched] 台股 ML 三相標記與多時間級別報告作業 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    
    # 1. 批次更新歷史行情快取 (加入重試與防 429 機制)
    run_step_with_retry(
        "python3 ~/stock-analysis/scripts/taiex_rsi_divergence.py --update --restore-weight",
        "批次更新歷史行情快取"
    )
    
    # 休息 3 秒避免 API 頻繁呼叫
    time.sleep(3)
    
    # 2. 生成最新 ML 標籤與定常特徵 CSV
    run_step_with_retry(
        "python3 ~/stock-analysis/scripts/generate_ml_labels.py",
        "生成 ML 標籤與定常特徵 CSV"
    )
    
    # 3. 檢查資料完整性與時效性
    csv_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
        
    latest_date_str = None
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            latest_date_str = row["Date"]
            
    if not latest_date_str:
        raise ValueError("無法讀取 CSV 資料日期，資料可能毀損。")
        
    latest_dt = datetime.strptime(latest_date_str, "%Y-%m-%d")
    days_old = (datetime.now() - latest_dt).days
    print(f"最新資料日期: {latest_date_str} (距今 {days_old} 天)")
    
    if days_old > 5:
        error_msg = f"【警告】資料過期！最新資料日期為 {latest_date_str}，距今超過 5 天。作業中斷。"
        send_email("harry2281996@gmail.com", "[Batched] 【失敗警報】台股 ML 報告 - 資料過期警報", error_msg)
        sys.exit(1)
        
    # 4. 執行模型訓練與推理 (分段批次)
    run_step_with_retry(
        "python3 ~/stock-analysis/scripts/train_taiex_tablet_model.py",
        "執行模型訓練與多時間級別推理"
    )
    
    # 5. 讀取推理報告
    report_path = os.path.expanduser("~/stock-analysis/models/taiex_tablet_models_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    # 6. 組裝 Email 內文
    subject = f"[Batched] 【台股 ML 盤勢報告】^TWII 多時間級別 (1d/3d/5d) 三相預測 - {latest_date_str}"
    
    body = f"""==================================================
       台股加權指數 (^TWII) 機器學習每日三相預測報告 (Batched 測試版)
==================================================
最新資料日期 : {latest_date_str}
模型訓練平臺 : Xiaomi Pad 7 (Termux - Batched Pipeline)
特徵工程架構 : 100% 定常化 (燭體相對比例 + 動態量比 + 乖離率 + 擺盪指標)

--------------------------------------------------
【多時間級別 (1/3/5 日後) 盤勢預測摘要】
--------------------------------------------------
"""

    for horizon, data in report.items():
        h_name = data["horizon"]
        pred = data["latest_prediction"]
        acc = data["test_accuracy"]
        conf = data["latest_confidence"]
        
        body += f"▶ 【預測未來 {h_name} 盤勢】: 【 {pred} 】\n"
        body += f"   - Out-of-Sample 測試集準率 : {acc * 100:.2f}%\n"
        body += f"   - 各狀態信心度比率 :\n"
        body += f"       • 上漲 (Up)           : {conf.get('Up (1)', 0)*100:6.2f}%\n"
        body += f"       • 盤整 (Consolidation) : {conf.get('Consolidation (0)', 0)*100:6.2f}%\n"
        body += f"       • 下跌 (Down)         : {conf.get('Down (-1)', 0)*100:6.2f}%\n\n"
        
    body += """--------------------------------------------------
【模型操作建議與說明】
--------------------------------------------------
• 1 日預測：適用於極短線部位或當沖/隔日沖避險參考。
• 3 日預測：適用於短線波段部位進退場判斷。
• 5 日預測：適用於一週短波段趨勢對齊。
• 若預測為【盤整】，建議控制部位曝險比例 (例如 0.3x ~ 0.5x) 或進行低吸高拋區間操作。

==================================================
附件說明：taiex_ml_training_data.csv 為包含歷史全特徵與雙重門檻 ZigZag 標籤之最新完整資料集。
==================================================
"""

    print("發送 Email 報告...")
    send_email("harry2281996@gmail.com", subject, body, attachment_path=csv_path)
    print("=== Batched 排程報告作業順利完成 ===")

if __name__ == "__main__":
    main()
