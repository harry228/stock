#!/usr/bin/env python3
"""
USD/TWD Momentum Model Report & Cron Job (美元兌新台幣動能模型報告與排程寄送)
- Runs usdtwd_momentum_model.py
- Captures output report
- Emails the report, backtest CSV, and analysis chart to neolin909@gmail.com
- Syncs the analysis chart, backtest CSV, and combined data CSV to Google Drive (gdrive:stock-analysis/)
"""

import os
import sys
import subprocess
from datetime import datetime

# Add home directory to path to import send_email
sys.path.append(os.path.expanduser("~"))
from send_email import send_email

def main():
    print(f"[{datetime.now()}] 開始執行 USD/TWD 匯率動能模型與報告排程作業...")
    
    # 1. 執行 USD/TWD 動能分析腳本並擷取終端輸出
    cmd = [
        sys.executable,
        os.path.expanduser("~/stock-analysis/scripts/usdtwd_momentum_model.py")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode != 0:
        print(f"USD/TWD 動能模型執行失敗！錯誤資訊: {result.stderr}")
        return
        
    output_text = result.stdout
    print(output_text)
    
    # 2. 擷取報告本文
    report_marker = "【USD/TWD】美元兌新台幣動能效率與多維度對稱模型分析報告"
    report_content = ""
    if report_marker in output_text:
        parts = output_text.split(report_marker)
        report_content = f"【USD/TWD】{report_marker}{parts[-1]}"
    else:
        report_content = output_text

    # 3. 寄送電子郵件並附上報表與圖表
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"USD/TWD 美元兌新台幣動能與多維度對稱分析報告 ({today_str})"
    
    csv_path = os.path.expanduser("~/storage/downloads/usdtwd_momentum_backtest.csv")
    img_path = os.path.expanduser("~/storage/downloads/usdtwd_momentum_analysis.png")
    combined_csv = os.path.expanduser("~/stock-analysis/data/usdtwd_combined_data.csv")
    
    print("正在發送電子郵件...")
    attachments = [img_path, csv_path]
    send_email(
        to_email="neolin909@gmail.com",
        subject=subject,
        body=report_content,
        attachment_path=attachments
    )
    print("郵件發送完成！")
    
    # 4. 自動同步至雲端硬碟 (gdrive:stock-analysis/)
    print("正在同步報告、圖表與數據 CSV 至 Google Drive (gdrive:stock-analysis/)...")
    sync_files = [img_path, csv_path, combined_csv]
    for f in sync_files:
        if os.path.exists(f):
            try:
                subprocess.run(['rclone', 'copy', f, 'gdrive:stock-analysis/'], check=True)
                print(f"成功同步: {os.path.basename(f)}")
            except subprocess.CalledProcessError as e:
                print(f"同步 {os.path.basename(f)} 失敗: {e}")
        else:
            print(f"檔案不存在，跳過同步: {f}")

if __name__ == "__main__":
    main()
