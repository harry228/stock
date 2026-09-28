#!/usr/bin/env python3
"""
台股波段高低點轉折與漲跌幅分析 (ZigZag Algorithm)
- 核心邏輯：使用高低點。若跌幅/漲幅超過指定門檻（預設 3%），視為有效轉折，並繪製波段轉折圖。
- 資料來源：~/stock-analysis/data/TWII_cache.json (台灣加權指數 ^TWII)
"""

import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 設定繁體中文顯示 (針對 Termux / Android 常見字型)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="台股波段高低點轉折分析")
    parser.add_argument(
        "--symbol", type=str, default="^TWII",
        help="股票/指數代號（預設: ^TWII）"
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="轉折門檻百分比（預設: 3.0，即 3%%）"
    )
    parser.add_argument(
        "--days", type=int, default=500,
        help="繪圖呈現的最近交易日天數（預設: 500 天，避免圖表過於擁擠）"
    )
    parser.add_argument(
        "--output-img", type=str,
        default=os.path.expanduser("~/storage/downloads/taiex_zigzag.png"),
        help="轉折圖儲存路徑"
    )
    parser.add_argument(
        "--output-csv", type=str,
        default=os.path.expanduser("~/storage/downloads/taiex_wave_summary.csv"),
        help="波段數據 CSV 儲存路徑"
    )
    return parser.parse_args()

def load_data(symbol):
    """載入快取資料"""
    clean_symbol = symbol.replace("^", "")
    cache_path = os.path.expanduser(f"~/stock-analysis/data/{clean_symbol}_cache.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"找不到快取檔案: {cache_path}，請先執行更新。")
    
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]

def calculate_zigzag(records, threshold_pct=3.0):
    """
    計算 ZigZag 波段轉折
    - 使用 High/Low 進行偵測
    - threshold_pct: 轉折門檻，例如 3.0 代表 3%
    """
    threshold = threshold_pct / 100.0
    dates = [r["date"] for r in records]
    highs = [r["high"] for r in records]
    lows = [r["low"] for r in records]
    closes = [r["close"] for r in records]
    
    n = len(records)
    if n == 0:
        return []
    
    pivots = []
    
    # 初始狀態判定
    direction = 0  # 1: 尋找波峰(UP), -1: 尋找波谷(DOWN)
    extreme_val = closes[0]
    extreme_idx = 0
    
    # 第一點預設為起始點
    pivots.append({
        "date": dates[0],
        "type": "start",
        "value": closes[0],
        "index": 0
    })
    
    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        
        if direction == 0:
            # 尚未建立方向，尋找第一個突破
            if h > extreme_val:
                extreme_val = h
                extreme_idx = i
            
            # 若自起點下跌超過 3%
            if l <= pivots[0]["value"] * (1 - threshold):
                # 確立前點為波峰，進入尋找波谷(DOWN)狀態
                pivots[0]["type"] = "peak"
                # 重新尋找從 0 到 i 之間的最大 High 作為波峰
                peak_idx = 0
                peak_val = highs[0]
                for k in range(1, i + 1):
                    if highs[k] > peak_val:
                        peak_val = highs[k]
                        peak_idx = k
                pivots[0]["index"] = peak_idx
                pivots[0]["value"] = peak_val
                pivots[0]["date"] = dates[peak_idx]
                
                direction = -1
                extreme_val = l
                extreme_idx = i
                
            # 若自起點上漲超過 3%
            elif h >= pivots[0]["value"] * (1 + threshold):
                # 確立前點為波谷，進入尋找波峰(UP)狀態
                pivots[0]["type"] = "trough"
                # 重新尋找從 0 到 i 之間的最小 Low 作為波谷
                trough_idx = 0
                trough_val = lows[0]
                for k in range(1, i + 1):
                    if lows[k] < trough_val:
                        trough_val = lows[k]
                        trough_idx = k
                pivots[0]["index"] = trough_idx
                pivots[0]["value"] = trough_val
                pivots[0]["date"] = dates[trough_idx]
                
                direction = 1
                extreme_val = h
                extreme_idx = i
                
        elif direction == 1:
            # 尋找波峰 (UP)
            if h > extreme_val:
                extreme_val = h
                extreme_idx = i
            
            # 若從波峰回檔跌幅超過 3%
            if l <= extreme_val * (1 - threshold):
                # 確立波峰
                pivots.append({
                    "date": dates[extreme_idx],
                    "type": "peak",
                    "value": extreme_val,
                    "index": extreme_idx
                })
                # 轉為尋找波谷 (DOWN)
                direction = -1
                extreme_val = l
                extreme_idx = i
                
        elif direction == -1:
            # 尋找波谷 (DOWN)
            if l < extreme_val:
                extreme_val = l
                extreme_idx = i
                
            # 若從波谷反彈漲幅超過 3%
            if h >= extreme_val * (1 + threshold):
                # 確立波谷
                pivots.append({
                    "date": dates[extreme_idx],
                    "type": "trough",
                    "value": extreme_val,
                    "index": extreme_idx
                })
                # 轉為尋找波峰 (UP)
                direction = 1
                extreme_val = h
                extreme_idx = i
                
    # 將最後一天的收盤價作為結束點
    pivots.append({
        "date": dates[-1],
        "type": "end",
        "value": closes[-1],
        "index": n - 1
    })
    
    return pivots

def generate_waves_summary(pivots, records):
    """
    根據轉折點生成波段統計摘要
    """
    waves = []
    for i in range(len(pivots) - 1):
        p1 = pivots[i]
        p2 = pivots[i+1]
        
        start_date = p1["date"]
        end_date = p2["date"]
        start_val = p1["value"]
        end_val = p2["value"]
        
        change_val = end_val - start_val
        change_pct = (change_val / start_val) * 100.0
        
        # 計算實際日曆天數與交易日天數
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days = (d2 - d1).days
        trading_days = p2["index"] - p1["index"]
        
        # 波段屬性
        if change_pct > 0:
            wave_type = "波段上漲"
        else:
            wave_type = "波段下跌"
            
        waves.append({
            "No": i + 1,
            "wave_type": wave_type,
            "start_date": start_date,
            "end_date": end_date,
            "start_val": round(start_val, 2),
            "end_val": round(end_val, 2),
            "change_val": round(change_val, 2),
            "change_pct": round(change_pct, 2),
            "trading_days": trading_days,
            "calendar_days": calendar_days
        })
    return waves

def save_to_csv(waves, output_path):
    """儲存波段摘要至 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import csv
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "No", "wave_type", "start_date", "end_date", "start_val", "end_val", "change_val", "change_pct", "trading_days", "calendar_days"
        ])
        writer.writeheader()
        writer.writerows(waves)
    print(f"波段統計 CSV 已儲存至: {output_path}")

def plot_zigzag(records, pivots, days_to_show, output_path, symbol, threshold_pct):
    """
    繪製波段轉折圖 (使用英文標籤，避免 Termux 中文字型缺失警告)
    """
    # 篩選要顯示的資料區間
    recent_records = records[-days_to_show:]
    start_date_str = recent_records[0]["date"]
    
    # 篩選對應時間區間內的 pivots
    start_idx_all = len(records) - days_to_show
    plot_pivots = [p for p in pivots if p["index"] >= start_idx_all]
    
    # 若首個 pivot 不是該區間的起始，在最前面加上一個起點，使線條完整
    if not plot_pivots or plot_pivots[0]["index"] > start_idx_all:
        plot_pivots.insert(0, {
            "date": start_date_str,
            "type": "start",
            "value": recent_records[0]["close"],
            "index": start_idx_all
        })
        
    # 準備繪圖數據
    dates_raw = [datetime.strptime(r["date"], "%Y-%m-%d") for r in recent_records]
    closes = [r["close"] for r in recent_records]
    
    pivot_dates = [datetime.strptime(p["date"], "%Y-%m-%d") for p in plot_pivots]
    pivot_values = [p["value"] for p in plot_pivots]
    
    plt.figure(figsize=(15, 8))
    
    # 繪製加權指數收盤價
    plt.plot(dates_raw, closes, color="lightgray", label=f"{symbol} Close", alpha=0.7, linewidth=1)
    
    # 繪製波段轉折線 (ZigZag)
    plt.plot(pivot_dates, pivot_values, color="royalblue", label=f"ZigZag Line (>{threshold_pct}%)", linewidth=2.5, linestyle="-")
    
    # 標記高低轉折點
    for p, p_date, p_val in zip(plot_pivots, pivot_dates, pivot_values):
        if p["type"] == "peak":
            plt.scatter(p_date, p_val, color="crimson", s=80, zorder=5, marker="^")
            plt.annotate(f"Peak\n{p_val:.1f}", (p_date, p_val), textcoords="offset points", xytext=(0,10), ha='center', color="crimson", fontweight="bold")
        elif p["type"] == "trough":
            plt.scatter(p_date, p_val, color="forestgreen", s=80, zorder=5, marker="v")
            plt.annotate(f"Trough\n{p_val:.1f}", (p_date, p_val), textcoords="offset points", xytext=(0,-20), ha='center', color="forestgreen", fontweight="bold")
            
    # 在線條中間標記漲跌幅百分比
    for i in range(len(plot_pivots) - 1):
        p1 = plot_pivots[i]
        p2 = plot_pivots[i+1]
        
        d1 = datetime.strptime(p1["date"], "%Y-%m-%d")
        d2 = datetime.strptime(p2["date"], "%Y-%m-%d")
        mid_date = d1 + (d2 - d1) / 2
        mid_val = p1["value"] + (p2["value"] - p1["value"]) / 2
        
        pct = ((p2["value"] - p1["value"]) / p1["value"]) * 100.0
        color = "crimson" if pct > 0 else "forestgreen"
        sign = "+" if pct > 0 else ""
        
        plt.annotate(f"{sign}{pct:.1f}%", (mid_date, mid_val), textcoords="offset points", xytext=(0,5), ha='center', color=color, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3, ec="none"))

    plt.title(f"{symbol} Trend Swings & Pivot Points (Threshold: {threshold_pct}%)", fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price / Index Point", fontsize=12)
    
    # 格式化 X 軸日期顯示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"波段轉折圖已儲存至: {output_path}")

def main():
    args = parse_args()
    
    try:
        records = load_data(args.symbol)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        return
        
    pivots = calculate_zigzag(records, args.threshold)
    waves = generate_waves_summary(pivots, records)
    
    # 儲存 CSV 檔案
    save_to_csv(waves, args.output_csv)
    
    # 繪製折線圖
    plot_zigzag(records, pivots, args.days, args.output_img, args.symbol, args.threshold)
    
    # 印出最近的 10 個波段轉折摘要
    print("\n" + "="*85)
    print(f"               {args.symbol} 最近 10 個波段轉折摘要 (門檻: {args.threshold}%)")
    print("="*85)
    print(f"{'No':<4}{'波段屬性':<10}{'起點日期':<12}{'終點日期':<12}{'起點值':<10}{'終點值':<10}{'漲跌幅(%)':<12}{'交易日':<6}")
    print("-"*85)
    
    # 取最近 10 筆波段
    recent_waves = waves[-10:]
    for w in recent_waves:
        sign = "+" if w["change_pct"] > 0 else ""
        print(f"{w['No']:<4}{w['wave_type']:<10}{w['start_date']:<12}{w['end_date']:<12}{w['start_val']:<10.2f}{w['end_val']:<10.2f}{sign+str(w['change_pct'])+'%':<12}{w['trading_days']:<6}")
    print("="*85)

if __name__ == "__main__":
    main()
