#!/usr/bin/env python3
"""
Soybean Futures Prediction & Report Job (黃豆期貨量價動能預報與郵件寄送排程)
- Runs soybean_momentum_model.py
- Captures output report
- Emails the completed report with CSV and PNG attachments to neolin909@gmail.com
"""

import os
import sys
import subprocess
from datetime import datetime

# Add home directory to path to import send_email
sys.path.append(os.path.expanduser("~"))
from send_email import send_email

def main():
    print(f"[{datetime.now()}] 開始執行黃豆期貨動能模型與報告發送作業...")
    
    # 1. 執行黃豆期貨動能分析腳本並擷取終端輸出
    cmd = [
        sys.executable,
        os.path.expanduser("~/stock-analysis/scripts/soybean_momentum_model.py")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode != 0:
        print(f"動能模型執行失敗！錯誤資訊: {result.stderr}")
        return
        
    output_text = result.stdout
    print(output_text)
    
    # 2. 擷取報告本文
    report_marker = "黃豆期貨量價效率與波動對稱動能模型報告"
    report_content = ""
    if report_marker in output_text:
        # 擷取 marker 開始到最後
        parts = output_text.split(report_marker)
        report_content = f"【ZS=F】{report_marker}{parts[-1]}"
    else:
        report_content = output_text

    # 3. 寄送電子郵件並附上報表與圖表
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"黃豆期貨動能效率與波動對稱分析報告 ({today_str})"
    
    csv_path = os.path.expanduser("~/storage/downloads/soybean_momentum_backtest.csv")
    img_path = os.path.expanduser("~/storage/downloads/soybean_momentum_analysis.png")
    
    # 圖表作為主要附件發送
    print("正在發送電子郵件...")
    send_email(
        to_email="neolin909@gmail.com",
        subject=subject,
        body=report_content,
        attachment_path=img_path
    )
    print("郵件發送完成！")

    # 4. 自動同步至雲端硬碟 (gdrive:Soybean)
    print("正在同步報告至 Google Drive (Soybean)...")
    for f in [csv_path, img_path]:
        if os.path.exists(f):
            try:
                subprocess.run(['rclone', 'copy', f, 'gdrive:Soybean/'], check=True)
                print(f"成功同步: {os.path.basename(f)}")
            except subprocess.CalledProcessError as e:
                print(f"同步 {os.path.basename(f)} 失敗: {e}")

if __name__ == "__main__":
    main()
