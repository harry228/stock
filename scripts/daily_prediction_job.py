#!/usr/bin/env python3
"""
Daily Prediction Job (每日量價動能預測報告排程 - 重構版)
- Automatically updates ^TWII cache and runs the Dynamic Momentum Model.
- Captures output from the main model script.
- Emails the report to neolin909@gmail.com with attachments.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add home directory to path to import send_email
sys.path.append(os.path.expanduser("~"))
from send_email import send_email

def main():
    print(f"[{datetime.now()}] 開始執行每日台股量價動能報告作業...")
    
    # 1. 呼叫主模型腳本 (含更新與還原權值)
    model_script = os.path.expanduser("~/stock-analysis/scripts/dynamic_momentum_model.py")
    cmd = [
        sys.executable,
        model_script,
        "--symbol", "^TWII",
        "--update",
        "--restore-weight"
    ]
    
    print("正在執行量價動能模型計算...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode != 0:
        print(f"模型執行失敗！錯誤資訊: {result.stderr}")
        return
        
    output_text = result.stdout
    
    # 2. 擷取報告本文
    report_marker = "【^TWII】量價效率與波動對稱動能模型"
    report_content = ""
    if report_marker in output_text:
        parts = output_text.split(report_marker)
        report_content = f"【^TWII】{report_marker}{parts[-1]}"
    else:
        report_content = output_text

    # 3. 強制等待檔案同步 (防止附件抓到舊檔)
    time.sleep(2)

    # 4. 寄送郵件
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"台股量價動能趨勢預估報告 ({today_str})"
    img_path = os.path.expanduser("~/storage/downloads/market_momentum_analysis.png")
    
    print("正在發送電子郵件...")
    send_email(
        to_email="neolin909@gmail.com",
        subject=subject,
        body=report_content,
        attachment_path=img_path
    )
    print("台股動能報告發送完成！")

    # 同步至雲端
    print("正在同步附件至雲端...")
    csv_path = os.path.expanduser("~/storage/downloads/momentum_prediction_backtest.csv")
    for f in [img_path, csv_path]:
        if os.path.exists(f):
            subprocess.run(["rclone", "copy", f, "gdrive:stock-analysis/"], check=True)
            print(f"成功同步: {os.path.basename(f)}")

if __name__ == "__main__":
    main()
