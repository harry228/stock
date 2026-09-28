import os
import csv
import glob
from datetime import datetime

def check_daily_signals():
    data_dir = os.path.expanduser("~/stock-analysis/data")
    today = datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.expanduser(f"~/stock-analysis/data/daily_rsi_signals_{today}.csv")
    
    # 讀取 rsi_*.csv 檔案
    files = glob.glob(os.path.join(data_dir, "rsi_*.csv"))
    
    with open(report_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["日期", "標的", "訊號(1買/-1賣)", "RSI14", "收盤價"])
        
        for file in files:
            symbol = os.path.basename(file).replace("rsi_", "").replace(".csv", "")
            with open(file, "r", encoding="utf-8-sig") as csvf:
                # 使用 DictReader 處理標題
                reader = csv.DictReader(csvf)
                rows = list(reader)
                if not rows: continue
                
                # 取最後一筆資料（最新交易日）
                last_row = rows[-1]
                signal = last_row.get("訊號")
                
                # 排除空值或 0 的訊號，只記錄有意義的背離訊號
                if signal and signal.strip() not in ["", "0"]:
                    writer.writerow([
                        last_row["日期"],
                        symbol,
                        signal,
                        last_row["RSI14"],
                        last_row["收盤"]
                    ])
                    
    print(f"每日背離訊號匯總已產出: {report_file}")

if __name__ == "__main__":
    check_daily_signals()
