import pandas as pd
import os
import sys
import subprocess

def check_integrity(csv_path):
    if not os.path.exists(csv_path):
        print(f"檔案不存在: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if len(df) < 25:
            print(f"警報: 報告筆數過少 ({len(df)})")
            return False
        if df['RSI14'].isnull().sum() > 5:
            print("警報: RSI14 欄位異常空值過多")
            return False
        return True
    except Exception as e:
        print(f"檢查過程發生錯誤: {e}")
        return False

def trigger_retry(job_name):
    print(f"正在嘗試重新觸發: {job_name}")
    # 這裡可以透過 hermes 或直接執行對應的 script
    # 範例邏輯：呼叫 RSI 每日報告腳本
    cmd = ["python3", os.path.expanduser("~/stock-analysis/scripts/daily_report_job.py")]
    subprocess.run(cmd)

if __name__ == "__main__":
    report_path = os.path.expanduser("~/storage/downloads/類股_RSI背離訊號匯總.csv")
    if not check_integrity(report_path):
        print("完整性檢查失敗，正在啟動補寄程序...")
        trigger_retry("RSI 每日報告")
        sys.exit(1)
    else:
        print("檢查通過: 報告數據完整。")
