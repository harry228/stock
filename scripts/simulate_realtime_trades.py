import json
import csv
import os

def simulate_realtime_zigzag(symbol="GLD", cache_file=None, threshold_pct=10.0, output_csv=None):
    if cache_file is None:
        cache_file = os.path.expanduser(f"~/stock-analysis/data/{symbol}_cache.json")
    if output_csv is None:
        output_csv = os.path.expanduser(f"~/storage/downloads/{symbol}_realtime_system_trades.csv")
        
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        records = data["records"]
        
    threshold = threshold_pct / 100.0
    
    # 逐日推演變數
    direction = 0  # 1: 當前判定為多頭/上漲, -1: 當前判定為空頭/下跌
    extreme_val = records[0]["close"]
    extreme_date = records[0]["date"]
    
    trades = []
    current_position = None # {"type": "Long"/"Short", "signal_date": ..., "entry_price": ..., "peak_date": ...}
    
    for i, r in enumerate(records):
        d = r["date"]
        h = r["high"]
        l = r["low"]
        c = r["close"]
        
        if direction == 0:
            if h > extreme_val:
                extreme_val = h
                extreme_date = d
            if l <= extreme_val * (1 - threshold):
                # 觸發做空訊號
                direction = -1
                extreme_val = l
                extreme_date = d
                current_position = {
                    "波段類型": "空頭(放空)",
                    "回溯轉折高點日": records[0]["date"],
                    "回溯高點價": records[0]["close"],
                    "即時觸發進場日(發信日)": d,
                    "進場價格(收盤價)": c
                }
            elif h >= extreme_val * (1 + threshold):
                # 觸發做多訊號
                direction = 1
                extreme_val = h
                extreme_date = d
                current_position = {
                    "波段類型": "多頭(做多)",
                    "回溯轉折低點日": records[0]["date"],
                    "回溯低點價": records[0]["close"],
                    "即時觸發進場日(發信日)": d,
                    "進場價格(收盤價)": c
                }
        elif direction == 1: # 當前為多頭持倉
            if h > extreme_val:
                extreme_val = h
                extreme_date = d
            # 檢查是否自波段最高點拉回超過 threshold (10%)
            if l <= extreme_val * (1 - threshold) or c <= extreme_val * (1 - threshold):
                # 平多倉並反手做空
                exit_date = d
                exit_price = c
                entry_price = current_position["進場價格(收盤價)"]
                profit_pct = round((exit_price - entry_price) / entry_price * 100, 2)
                
                trades.append({
                    "波段類型": current_position["波段類型"],
                    "回溯波段起點日": current_position.get("回溯轉折低點日", ""),
                    "即時進場發信日": current_position["即時觸發進場日(發信日)"],
                    "實際進場價": entry_price,
                    "即時出場發信日": exit_date,
                    "實際出場價": exit_price,
                    "實際獲利(%)": profit_pct,
                    "持倉天數": (i - [x["date"] for x in records].index(current_position["即時觸發進場日(發信日)"])),
                    "最高浮動利潤點日": extreme_date,
                    "最高浮動價格": extreme_val
                })
                
                # 反手做空
                direction = -1
                extreme_val = l
                extreme_date = d
                current_position = {
                    "波段類型": "空頭(放空)",
                    "回溯轉折高點日": extreme_date,
                    "回溯高點價": extreme_val,
                    "即時觸發進場日(發信日)": d,
                    "進場價格(收盤價)": c
                }
        elif direction == -1: # 當前為空頭持倉
            if l < extreme_val:
                extreme_val = l
                extreme_date = d
            # 檢查是否自波段最低點反彈超過 threshold (10%)
            if h >= extreme_val * (1 + threshold) or c >= extreme_val * (1 + threshold):
                # 平空倉並反手做多
                exit_date = d
                exit_price = c
                entry_price = current_position["進場價格(收盤價)"]
                profit_pct = round((entry_price - exit_price) / entry_price * 100, 2)
                
                trades.append({
                    "波段類型": current_position["波段類型"],
                    "回溯波段起點日": current_position.get("回溯轉折高點日", ""),
                    "即時進場發信日": current_position["即時觸發進場日(發信日)"],
                    "實際進場價": entry_price,
                    "即時出場發信日": exit_date,
                    "實際出場價": exit_price,
                    "實際獲利(%)": profit_pct,
                    "持倉天數": (i - [x["date"] for x in records].index(current_position["即時觸發進場日(發信日)"])),
                    "最低浮動利潤點日": extreme_date,
                    "最低浮動價格": extreme_val
                })
                
                # 反手做多
                direction = 1
                extreme_val = h
                extreme_date = d
                current_position = {
                    "波段類型": "多頭(做多)",
                    "回溯轉折低點日": extreme_date,
                    "回溯低點價": extreme_val,
                    "即時觸發進場日(發信日)": d,
                    "進場價格(收盤價)": c
                }

    # 輸出 CSV
    fieldnames = [
        "波段類型", "回溯波段起點日", "即時進場發信日", "實際進場價", 
        "即時出場發信日", "實際出場價", "實際獲利(%)", "持倉天數"
    ]
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
        
    print(f"已產出即時系統交易回測: {output_csv} (共 {len(trades)} 筆交易)")
    return trades

if __name__ == "__main__":
    # 執行黃金 (GLD) 與台股加權指數 (TWII)
    simulate_realtime_zigzag("GLD", threshold_pct=10.0, output_csv=os.path.expanduser("~/storage/downloads/gold_realtime_system_trades.csv"))
    simulate_realtime_zigzag("TWII", threshold_pct=10.0, output_csv=os.path.expanduser("~/storage/downloads/taiex_realtime_system_trades.csv"))
