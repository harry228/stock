import os
import subprocess
import csv
from datetime import datetime
import shutil

def send_report():
    # 1. 取得最新資料日期
    data_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(data_path):
        data_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
    
    latest_date_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))
            if reader:
                latest_date_str = reader[-1]["Date"]

    chart_path = os.path.expanduser("~/storage/downloads/taiex_forecast_chart.png")
    
    # 2. 備份並添加日期後綴
    print("準備備份圖表...")
    if os.path.exists(chart_path):
        date_suffix = latest_date_str.replace("-", "")
        backup_name = f"taiex_forecast_chart_{date_suffix}.png"
        backup_path = os.path.join(os.path.expanduser("~/Backup"), backup_name)
        
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(chart_path, backup_path)
        
        # 同步至雲端
        subprocess.run(["rclone", "copy", backup_path, "gdrive:Backup/"], check=True)
        print(f"備份至 {backup_path} 並同步至雲端完成。")
    
    # 3. 發送郵件
    subject = f"[PREDICTION] TAIEX MULTI-TASK ENSEMBLE FUTURE MARKET OUTLOOK - {latest_date_str}"
    email_cmd = [
        "python3", os.path.expanduser("~/send_email.py"),
        "--to", "neolin909@gmail.com",
        "--subject", subject,
        "--body", f"台股多任務集成預測報告 (數據日期: {latest_date_str})，附件為未來 5 日市場預測展望圖表。"
    ]
    if os.path.exists(chart_path):
        email_cmd.extend(["--file", chart_path])
    
    try:
        subprocess.run(email_cmd, check=True)
        print("郵件發送成功！")
    except Exception as e:
        print(f"郵件發送失敗: {e}")

if __name__ == "__main__":
    send_report()
