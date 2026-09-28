#!/usr/bin/env python3
"""
週與月 KD 指標分析報告作業 (含蒙地卡羅模擬與 Email 寄送)
- 邏輯參考 RSI 報告作業
- 包含 KD 偵測、蒙地卡羅驗證、郵件發送
"""

import subprocess
import os
import sys
import json
import argparse
from datetime import datetime

# 路徑設定
BASE_DIR = os.path.expanduser("~/stock-analysis")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
SEND_EMAIL_SCRIPT = os.path.expanduser("~/send_email.py")

def run_command(cmd):
    try:
        # 使用 subprocess.list 型式避免 shell=True 的引號問題
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"執行失敗: {cmd}\n錯誤: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="^TWII,0050.TW")
    parser.add_argument("--timeframe", choices=['W', 'M'], default='W')
    args = parser.parse_args()

    tf_name = "週" if args.timeframe == 'W' else "月"
    print(f"[{datetime.now()}] 開始執行{tf_name} KD 報告作業...")

    # 1. 執行 KD 指標分析 (會更新資料與計算)
    print(f"1/3 正在執行 {tf_name} KD 分析...")
    kd_cmd = [
        "python3", os.path.join(SCRIPTS_DIR, "kd_indicator.py"),
        "--symbols", args.symbols,
        "--timeframe", args.timeframe
    ]
    kd_output = run_command(kd_cmd)
    if not kd_output:
        print("KD 分析失敗，中止後續作業。")
        return

    # 2. 執行 Monte Carlo 模擬
    # 注意：monte_carlo_bootstrap.py 目前主要針對 RSI，但我們可以其輸出的顯著性作為參考
    # 或者此處可擴展為 KD 專用的模擬，但目前先沿用通用的蒙地卡羅拔靴法流程
    print("2/3 正在執行 Monte Carlo 模擬風險評估...")
    mc_cmd = [
        "python3", os.path.join(SCRIPTS_DIR, "monte_carlo_bootstrap.py"),
        "--symbols", args.symbols
    ]
    mc_output = run_command(mc_cmd)
    
    # 組合內容
    summary_content = f"【{tf_name} KD 指標報告】\n{kd_output}\n"
    
    if mc_output:
        summary_marker = "多股對比及風控決策摘要表"
        if summary_marker in mc_output:
            mc_summary = mc_output.split(summary_marker)[-1]
            summary_content += f"\n【Monte Carlo 風險分析摘要】\n{summary_marker}{mc_summary}"
        else:
            summary_content += "\n(未能在 MC 輸出中找到摘要表)"
    else:
        summary_content += "\n(Monte Carlo 模擬執行失敗)"

    # 3. 寄送郵件
    print("3/3 正在發送郵件報告...")
    
    # 查找可能的產出 CSV (需確保 kd_indicator.py 有生成對應檔案)
    csv_path = os.path.expanduser(f"~/storage/downloads/{args.timeframe}_KD_report.csv")
    if os.path.exists(csv_path):
        print("正在同步附件至雲端...")
        run_command(f"rclone copy {csv_path} gdrive:stock-analysis/")
    
    subject = f"{tf_name} KD 指標與風險評估報告"
    
    email_cmd = [
        "python3", SEND_EMAIL_SCRIPT,
        "--subject", subject,
        "--body", summary_content
    ]
    
    # 若有 CSV 產出，可以附上 (此處假設 kd_indicator 目前僅 print)
    # 若未來 kd_indicator 支援輸出 CSV，可在此添加 --file
    
    try:
        subprocess.run(email_cmd, check=True)
        print(f"{tf_name} 報告作業完成！")
    except subprocess.CalledProcessError:
        print("郵件發送失敗。")

if __name__ == "__main__":
    main()
