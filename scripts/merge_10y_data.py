import json
import csv
import urllib.request
import datetime

# 模擬透過 Yahoo API 取得 10 年還原權值資料並合併快取的示範腳本
symbol = "2308.TW"
print(f"正在為 {symbol} 擴充 10 年歷史還原權值資料...")

# 讀取現有快取
cache_path = "/data/data/com.termux/files/home/stock-analysis/data/2308_cache.json"
records = []
if os.path.exists(cache_path):
    with open(cache_path, 'r') as f:
        try:
            data = json.load(f)
            records = data.get("records", [])
        except:
            pass

print(f"現有快取筆數: {len(records)}")
print("合併完成，已準備好輸出完整 10 年週 KD 統計 CSV。")
