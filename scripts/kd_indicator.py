#!/usr/bin/env python3
"""
週與月 KD 指標分析腳本
- 支援從 Yahoo Finance 抓取日線資料並轉換為週/月線
- 計算 KD 指標 (9, 3, 3)
- 輸出週與月 KD 交叉訊號
- 支援增量更新與還原權值
"""

import json
import csv
import os
import sys
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timedelta

DATA_DIR = os.path.expanduser("~/stock-analysis/data")
OUTPUT_DIR = os.path.expanduser("~/stock-analysis/output")

def fetch_yahoo_data(symbol, start_date=None, end_date=None, years=10):
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=years * 365 + 30)

    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
        f"&includeAdjustedClose=true"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception:
        try:
            url2 = url.replace("query1.finance", "query2.finance")
            req2 = urllib.request.Request(url2, headers=headers)
            with urllib.request.urlopen(req2, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return []

    if "chart" not in raw or not raw["chart"]["result"]:
        return []
        
    chart = raw["chart"]["result"][0]
    timestamps = chart.get("timestamp")
    if not timestamps:
        return []
        
    quote = chart["indicators"]["quote"][0]
    adjclose_data = chart["indicators"].get("adjclose")
    adjclose = adjclose_data[0].get("adjclose") if adjclose_data else quote["close"]

    records = []
    for i, ts in enumerate(timestamps):
        # 確保關鍵數值不為空
        if any(quote[k][i] is None for k in ["open", "high", "low", "close"]):
            continue
        if adjclose[i] is None:
            continue
            
        dt = datetime.fromtimestamp(ts)
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(quote["open"][i]), 2),
            "high": round(float(quote["high"][i]), 2),
            "low": round(float(quote["low"][i]), 2),
            "close": round(float(quote["close"][i]), 2),
            "adjclose": round(float(adjclose[i]), 2)
        })
    return records

def resample(records, timeframe='W'):
    resampled = []
    if not records: return []
    
    current_group = None
    group_data = []

    for r in records:
        dt = datetime.strptime(r["date"], "%Y-%m-%d")
        if timeframe == 'W':
            group_id = dt.isocalendar()[:2]
        else:
            group_id = (dt.year, dt.month)

        if group_id != current_group:
            if group_data:
                resampled.append(aggregate(group_data))
            current_group = group_id
            group_data = [r]
        else:
            group_data.append(r)
    
    if group_data:
        resampled.append(aggregate(group_data))
    
    return resampled

def aggregate(group_data):
    return {
        "date": group_data[-1]["date"],
        "open": group_data[0]["open"],
        "high": max(r["high"] for r in group_data),
        "low": min(r["low"] for r in group_data),
        "close": group_data[-1]["close"],
        "adjclose": group_data[-1]["adjclose"]
    }

def compute_kd(records, n=9, m1=3, m2=3):
    if len(records) < n:
        return []

    results = []
    k = 50.0
    d = 50.0

    for i in range(len(records)):
        if i < n - 1:
            results.append({"date": records[i]["date"], "k": None, "d": None})
            continue
        
        window = records[i-n+1 : i+1]
        high_n = max(r["high"] for r in window)
        low_n = min(r["low"] for r in window)
        close = records[i]["close"]
        
        if high_n == low_n:
            rsv = 50.0
        else:
            rsv = (close - low_n) / (high_n - low_n) * 100
        
        k = k * (m1 - 1) / m1 + rsv * (1 / m1)
        d = d * (m2 - 1) / m2 + k * (1 / m2)
        
        results.append({
            "date": records[i]["date"],
            "close": round(close, 2),
            "k": round(k, 2),
            "d": round(d, 2)
        })
    return results

def detect_signals(kd_results):
    signals = []
    for i in range(1, len(kd_results)):
        curr = kd_results[i]
        prev = kd_results[i-1]
        
        if curr["k"] is None or prev["k"] is None:
            continue
        
        if prev["k"] <= prev["d"] and curr["k"] > curr["d"]:
            signals.append({"date": curr["date"], "type": "黃金交叉", "k": curr["k"], "d": curr["d"], "close": curr["close"]})
        elif prev["k"] >= prev["d"] and curr["k"] < curr["d"]:
            signals.append({"date": curr["date"], "type": "死亡交叉", "k": curr["k"], "d": curr["d"], "close": curr["close"]})
            
    return signals

def process_symbol(symbol, args):
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_daily.json")
    
    daily_records = []
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                daily_records = json.load(f)
            except:
                daily_records = []
        
        if daily_records:
            last_date = daily_records[-1]["date"]
            new_data = fetch_yahoo_data(symbol, start_date=last_date)
            date_map = {r["date"]: r for r in daily_records}
            for r in new_data: date_map[r["date"]] = r
            daily_records = sorted(date_map.values(), key=lambda x: x["date"])
        else:
            daily_records = fetch_yahoo_data(symbol)
    else:
        daily_records = fetch_yahoo_data(symbol)
    
    if not daily_records:
        print(f"Skipping {symbol}: No data.")
        return

    with open(cache_file, "w") as f:
        json.dump(daily_records, f)

    # 還原權值處理
    is_stock = any(suffix in symbol for suffix in [".TW", ".TWO"])
    if args.restore_weight or is_stock:
        for r in daily_records:
            ratio = r["adjclose"] / r["close"] if r["close"] != 0 else 1.0
            r["open"] = round(r["open"] * ratio, 2)
            r["high"] = round(r["high"] * ratio, 2)
            r["low"] = round(r["low"] * ratio, 2)
            r["close"] = r["adjclose"]

    report_lines = [f"\n【{symbol} KD 指標分析報告】", f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if is_stock or args.restore_weight:
        report_lines.append("(已套用還原權值計算)")

    for tf_label, tf_code in [("週線", "W"), ("月線", "M")]:
        if args.timeframe == 'both' or args.timeframe == tf_code:
            resampled_data = resample(daily_records, tf_code)
            kd_results = compute_kd(resampled_data)
            if not kd_results: continue
            
            signals = detect_signals(kd_results)
            latest = kd_results[-1]
            prev = kd_results[-2] if len(kd_results) > 1 else None
            
            report_lines.append(f"\n[{tf_label} KD] 結束日期: {latest['date']}")
            report_lines.append(f"K: {latest['k']}, D: {latest['d']}")
            
            if signals and signals[-1]["date"] == latest["date"]:
                report_lines.append(f"🚩 訊號: {signals[-1]['type']}")
            elif prev:
                trend = "上升" if latest['k'] > prev['k'] else "下降"
                report_lines.append(f"動能趨勢: {trend}")
                
    print("\n".join(report_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="^TWII")
    parser.add_argument("--symbols", help="多個標的，以逗號分隔")
    parser.add_argument("--timeframe", choices=['W', 'M', 'both'], default='both')
    parser.add_argument("--restore-weight", action="store_true")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    symbols = args.symbols.split(",") if args.symbols else [args.symbol]
    for symbol in symbols:
        process_symbol(symbol, args)

if __name__ == "__main__":
    main()
