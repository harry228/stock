#!/usr/bin/env python3
"""
USD/TWD Multi-Source Incremental Data Fetcher (Stateful Chunking)
- Pure Python stdlib implementation (No pandas/numpy required for Termux)
- Incremental fetcher with anti-429 retry and progress persistence
- Data sources:
  1. Yahoo Finance (TWD=X, DX-Y.NYB, KRW=X, EURUSD=X, JPY=X, ^TNX, ^FVX)
  2. TWSE / FinMind (Foreign Institutional Investor Daily Net Flow)
"""

import os
import sys
import json
import csv
import time
import random
import datetime
import urllib.request
import urllib.error

# Add script dir to sys.path to import api_helper
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from api_helper import retry_on_429
except ImportError:
    def retry_on_429(max_attempts=4, initial_delay=15):
        def decorator(func):
            return func
        return decorator

DATA_DIR = os.path.expanduser("~/stock-analysis/data")
PROGRESS_FILE = os.path.join(DATA_DIR, "usdtwd_progress.json")
LOG_FILE = os.path.expanduser("~/stock-analysis/logs/usdtwd_fetch.log")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Define tasks to fetch
YFINANCE_ITEMS = {
    "usdtwd": {"symbol": "TWD=X", "desc": "美元兌新台幣匯率"},
    "dxy": {"symbol": "DX-Y.NYB", "desc": "美元指數"},
    "usdkrw": {"symbol": "KRW=X", "desc": "美元兌韓元匯率"},
    "eurusd": {"symbol": "EURUSD=X", "desc": "歐元兌美元匯率"},
    "usdjpy": {"symbol": "JPY=X", "desc": "美元兌日圓匯率"},
    "us10y": {"symbol": "^TNX", "desc": "美國 10 年期公債殖利率"},
    "us5y": {"symbol": "^FVX", "desc": "美國 5 年期公債殖利率"}
}

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"讀取進度檔失敗，重新建立: {e}")
    return {"completed_items": {}, "last_updated": None}

def save_progress(progress):
    progress["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

@retry_on_429(max_attempts=4, initial_delay=15)
def fetch_yfinance_symbol(symbol, years=10):
    """Fetch history from Yahoo Finance v8 chart API"""
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=years * 365)
    
    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
    }
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
        
    if "chart" not in raw or not raw["chart"]["result"]:
        log(f"Yahoo Finance 無法回傳 {symbol} 資料")
        return None
        
    chart = raw["chart"]["result"][0]
    timestamps = chart.get("timestamp", [])
    quote = chart["indicators"]["quote"][0]
    closes = quote.get("close", [])
    
    records = []
    for ts, close in zip(timestamps, closes):
        if close is not None:
            dt_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            records.append({"date": dt_str, "close": round(close, 4)})
            
    return records

@retry_on_429(max_attempts=4, initial_delay=15)
def fetch_twse_foreign_flow(start_date="2018-01-01"):
    """Fetch daily foreign investment net buy/sell from FinMind API"""
    url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockTotalInstitutionalInvestors&start_date={start_date}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    log(f"正從 FinMind API 下載三大法人買賣超歷史資料 (從 {start_date})...")
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
        
    data = raw.get("data", [])
    date_flow = {}
    
    for row in data:
        name = row.get("name", "")
        if "Foreign" in name or "外資" in name:
            dt = row.get("date")
            buy = row.get("buy", 0)
            sell = row.get("sell", 0)
            net = buy - sell
            if dt not in date_flow:
                date_flow[dt] = {"foreign_buy": 0, "foreign_sell": 0, "foreign_net": 0}
            date_flow[dt]["foreign_buy"] += buy
            date_flow[dt]["foreign_sell"] += sell
            date_flow[dt]["foreign_net"] += net
            
    records = []
    for dt in sorted(date_flow.keys()):
        records.append({
            "date": dt,
            "foreign_buy": date_flow[dt]["foreign_buy"],
            "foreign_sell": date_flow[dt]["foreign_sell"],
            "foreign_net": date_flow[dt]["foreign_net"]
        })
    return records

def process_item(item_key, info, progress):
    log(f"--> 開始處理 [{item_key}] ({info['desc']}: {info['symbol']})...")
    data = fetch_yfinance_symbol(info["symbol"], years=10)
    if data:
        out_file = os.path.join(DATA_DIR, f"usdtwd_{item_key}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        
        progress["completed_items"][item_key] = {
            "symbol": info["symbol"],
            "count": len(data),
            "latest_date": data[-1]["date"] if data else None,
            "status": "success",
            "file": out_file
        }
        save_progress(progress)
        log(f"✓ [{item_key}] 完成，共 {len(data)} 筆資料，最新日期: {data[-1]['date'] if data else 'N/A'}")
        return True
    else:
        log(f"✗ [{item_key}] 抓取失敗")
        return False

def process_twse_flow(progress):
    log("--> 開始處理 [twse_foreign_flow] (外資每日買賣超金額)...")
    data = fetch_twse_foreign_flow("2018-01-01")
    if data:
        out_file = os.path.join(DATA_DIR, "usdtwd_foreign_flow.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            
        progress["completed_items"]["twse_foreign_flow"] = {
            "count": len(data),
            "latest_date": data[-1]["date"] if data else None,
            "status": "success",
            "file": out_file
        }
        save_progress(progress)
        log(f"✓ [twse_foreign_flow] 完成，共 {len(data)} 筆資料，最新日期: {data[-1]['date'] if data else 'N/A'}")
        return True
    else:
        log("✗ [twse_foreign_flow] 抓取失敗")
        return False

def merge_datasets():
    log("--> 開始進行數據對齊與合併 (Data Alignment & Merging via Pure Python)...")
    progress = load_progress()
    completed = progress.get("completed_items", {})
    
    date_dict = {}
    all_cols = []
    
    # 1. YFinance items
    for item_key in YFINANCE_ITEMS:
        if item_key in completed:
            file_path = os.path.join(DATA_DIR, f"usdtwd_{item_key}.json")
            if os.path.exists(file_path):
                all_cols.append(item_key)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for row in data:
                    d = row["date"]
                    if d not in date_dict:
                        date_dict[d] = {}
                    date_dict[d][item_key] = row["close"]
                    
    # 2. TWSE Flow
    if "twse_foreign_flow" in completed:
        file_path = os.path.join(DATA_DIR, "usdtwd_foreign_flow.json")
        if os.path.exists(file_path):
            all_cols.append("foreign_net")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                d = row["date"]
                if d not in date_dict:
                    date_dict[d] = {}
                date_dict[d]["foreign_net"] = row["foreign_net"]
                
    if not date_dict:
        log("沒有任何已完成的數據可供合併。")
        return None
        
    sorted_dates = sorted(date_dict.keys())
    
    last_vals = {col: None for col in all_cols}
    output_rows = []
    
    for d in sorted_dates:
        row_data = {"date": d}
        for col in all_cols:
            val = date_dict[d].get(col)
            if val is not None:
                last_vals[col] = val
            row_data[col] = last_vals[col]
            
        if row_data.get("usdtwd") is not None:
            output_rows.append(row_data)
            
    output_csv = os.path.join(DATA_DIR, "usdtwd_combined_data.csv")
    fieldnames = ["date"] + all_cols
    
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        
    log(f"✓ 數據合併完成！寫入: {output_csv}，總筆數: {len(output_rows)}")
    return output_csv

def main():
    import argparse
    parser = argparse.ArgumentParser(description="USD/TWD Incremental Data Fetcher")
    parser.add_argument("--step-limit", type=int, default=2, help="單次執行的最大抓取項目數 (小步驟避免 429)")
    parser.add_argument("--merge-only", action="store_true", help="僅執行數據合併")
    args = parser.parse_args()
    
    if args.merge_only:
        merge_datasets()
        return
        
    progress = load_progress()
    completed = progress.get("completed_items", {})
    
    fetched_count = 0
    
    # 1. Process Yahoo Finance items step-by-step
    for item_key, info in YFINANCE_ITEMS.items():
        if item_key not in completed:
            success = process_item(item_key, info, progress)
            if success:
                fetched_count += 1
                sleep_time = random.uniform(3, 6)
                log(f"冷卻等待 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)
                
            if fetched_count >= args.step_limit:
                log(f"已達到單次執行限制 (--step-limit {args.step_limit})，自動停靠以防 429 限制。")
                break
                
    # 2. Process TWSE Foreign Flow if limit not reached
    if fetched_count < args.step_limit and "twse_foreign_flow" not in completed:
        process_twse_flow(progress)
        fetched_count += 1
        
    # Check total status
    total_tasks = len(YFINANCE_ITEMS) + 1
    completed_tasks = len(progress.get("completed_items", {}))
    log(f"目前總進度: {completed_tasks} / {total_tasks} 項目已完成。")
    
    if completed_tasks >= total_tasks:
        log("所有數據抓取任務已全數完成！開始執行合併...")
        merge_datasets()

if __name__ == "__main__":
    main()
