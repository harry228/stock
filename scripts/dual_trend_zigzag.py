#!/usr/bin/env python3
"""
Dual Trend ZigZag Classification & Analysis (Upward, Downward, Consolidation)
Combines 3% minor and 10% major ZigZag thresholds to segment market phases:
- Upward (上漲): Major 10% trend is up, not in consolidation.
- Downward (下跌): Major 10% trend is down, not in consolidation.
- Consolidation (盤整): Identified by successive 3% swings moving within a tight horizontal range (<= 8%) for >= 20 trading days.
"""

import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set font for Termux / Android environment
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="台股雙重趨勢盤整區分分析")
    parser.add_argument("--symbol", type=str, default="^TWII", help="股票/指數代號")
    parser.add_argument("--days", type=int, default=1200, help="圖表繪製天數")
    parser.add_argument("--range-pct", type=float, default=8.0, help="盤整波動上限百分比 (預設 8%%)")
    parser.add_argument("--min-days", type=int, default=20, help="最少盤整交易日天數 (預設 20 天)")
    parser.add_argument("--output-img", type=str, default=os.path.expanduser("~/storage/downloads/dual_trend_segments.png"))
    parser.add_argument("--output-csv", type=str, default=os.path.expanduser("~/storage/downloads/market_segments_summary.csv"))
    return parser.parse_args()

def calculate_zigzag(records, threshold_pct):
    threshold = threshold_pct / 100.0
    dates = [r["date"] for r in records]
    highs = [r["high"] for r in records]
    lows = [r["low"] for r in records]
    closes = [r["close"] for r in records]
    n = len(records)
    if n == 0: return []
    
    pivots = []
    direction = 0
    extreme_val = closes[0]
    extreme_idx = 0
    
    pivots.append({"date": dates[0], "type": "start", "value": closes[0], "index": 0})
    
    for i in range(1, n):
        h, l = highs[i], lows[i]
        if direction == 0:
            if h > extreme_val: extreme_val, extreme_idx = h, i
            if l <= pivots[0]["value"] * (1 - threshold):
                direction, extreme_val, extreme_idx = -1, l, i
            elif h >= pivots[0]["value"] * (1 + threshold):
                direction, extreme_val, extreme_idx = 1, h, i
        elif direction == 1:
            if h > extreme_val: extreme_val, extreme_idx = h, i
            if l <= extreme_val * (1 - threshold):
                pivots.append({"date": dates[extreme_idx], "type": "peak", "value": extreme_val, "index": extreme_idx})
                direction, extreme_val, extreme_idx = -1, l, i
        elif direction == -1:
            if l < extreme_val: extreme_val, extreme_idx = l, i
            if h >= extreme_val * (1 + threshold):
                pivots.append({"date": dates[extreme_idx], "type": "trough", "value": extreme_val, "index": extreme_idx})
                direction, extreme_val, extreme_idx = 1, h, i
                
    pivots.append({"date": dates[-1], "type": "end", "value": closes[-1], "index": n - 1})
    return pivots

def main():
    args = parse_args()
    clean_symbol = args.symbol.replace("^", "")
    cache_path = os.path.expanduser(f"~/stock-analysis/data/{clean_symbol}_cache.json")
    
    if not os.path.exists(cache_path):
        print(f"錯誤: 找不到快取檔案 {cache_path}")
        return
        
    with open(cache_path, "r", encoding="utf-8") as f:
        records = json.load(f)["records"]
        
    dates = [r["date"] for r in records]
    closes = [r["close"] for r in records]
    n = len(records)
    
    # 1. 計算兩組轉折點 (10% 大趨勢, 3% 中短波動)
    pivots_10 = calculate_zigzag(records, 10.0)
    pivots_3 = calculate_zigzag(records, 3.0)
    
    # 2. 定義初始日趨勢 (由 10% 趨勢決定)
    daily_state = []
    for i in range(n):
        state = "上漲"
        for j in range(len(pivots_10) - 1):
            p1 = pivots_10[j]
            p2 = pivots_10[j+1]
            if p1["index"] <= i <= p2["index"]:
                state = "上漲" if p2["value"] > p1["value"] else "下跌"
                break
        daily_state.append(state)
        
    # 3. 識別盤整期：3% 轉折點在指定交易日天數內波幅不大於 range_pct
    consolidations = []
    i = 0
    while i < len(pivots_3) - 1:
        best_j = -1
        for j in range(i + 2, len(pivots_3)):
            segment = pivots_3[i:j+1]
            vals = [p["value"] for p in segment]
            max_val = max(vals)
            min_val = min(vals)
            start_val = segment[0]["value"]
            
            # 若區間最大高低波幅在設定閾值內
            if (max_val - min_val) / start_val <= (args.range_pct / 100.0):
                trading_days = segment[-1]["index"] - segment[0]["index"]
                if trading_days >= args.min_days:
                    best_j = j
            else:
                break
                
        if best_j != -1:
            consolidations.append((pivots_3[i]["index"], pivots_3[best_j]["index"]))
            i = best_j  # 跳過盤整期
        else:
            i += 1
            
    # 4. 覆蓋盤整狀態
    for start, end in consolidations:
        for idx in range(start, end + 1):
            daily_state[idx] = "盤整"
            
    # 5. 合併連續狀態，生成區間清單
    final_waves = []
    curr_state = daily_state[0]
    start_idx = 0
    
    for i in range(1, n):
        if daily_state[i] != curr_state or i == n - 1:
            end_idx = i
            start_val = closes[start_idx]
            end_val = closes[end_idx]
            pct = (end_val - start_val) / start_val * 100
            days = end_idx - start_idx
            
            final_waves.append({
                "type": curr_state,
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "start_val": round(start_val, 2),
                "end_val": round(end_val, 2),
                "change_pct": round(pct, 2),
                "trading_days": days
            })
            curr_state = daily_state[i]
            start_idx = i
            
    # 6. 統計分析
    stats = {"上漲": [], "下跌": [], "盤整": []}
    for w in final_waves:
        stats[w["type"]].append(w)
        
    print("\n" + "="*80)
    print(f"             【{args.symbol}】上漲、盤整、下跌三相波段規律分析")
    print("="*80)
    
    csv_data = []
    for k, v in stats.items():
        count = len(v)
        if count > 0:
            avg_pct = sum([x["change_pct"] for x in v]) / count
            avg_days = sum([x["trading_days"] for x in v]) / count
            max_pct = max([x["change_pct"] for x in v])
            min_pct = min([x["change_pct"] for x in v])
            max_days = max([x["trading_days"] for x in v])
            min_days = min([x["trading_days"] for x in v])
            
            print(f"【{k}段】(共 {count} 次波段):")
            print(f"  - 平均漲跌幅: {avg_pct:+.2f}%  (區間: {min_pct:+.2f}% ~ {max_pct:+.2f}%)")
            print(f"  - 平均持續天數: {avg_days:.1f} 天 (區間: {min_days} ~ {max_days} 天)")
            print("-" * 50)
            
            csv_data.append({
                "類型": k,
                "次數": count,
                "平均漲跌幅(%)": round(avg_pct, 2),
                "最大漲跌幅(%)": round(max_pct, 2),
                "最小漲跌幅(%)": round(min_pct, 2),
                "平均交易日": round(avg_days, 1),
                "最大交易日": max_days,
                "最小交易日": min_days
            })
            
    # 寫入 CSV 檔案
    import csv
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["類型", "次數", "平均漲跌幅(%)", "最大漲跌幅(%)", "最小漲跌幅(%)", "平均交易日", "最大交易日", "最小交易日"])
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"數據分析報表已匯出至: {args.output_csv}")
    
    # 7. 繪圖 (使用 axvspan 填滿背景色區分狀態)
    recent_records = records[-args.days:]
    dates_raw = [datetime.strptime(r["date"], "%Y-%m-%d") for r in recent_records]
    recent_closes = [r["close"] for r in recent_records]
    start_date_str = recent_records[0]["date"]
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(dates_raw, recent_closes, color="#444444", label=f"{args.symbol} Close", linewidth=1.5, zorder=3)
    
    # 繪製背景色帶
    legend_added = {"上漲": False, "下跌": False, "盤整": False}
    for w in final_waves:
        if w["end_date"] < start_date_str:
            continue
            
        t_start = max(datetime.strptime(w["start_date"], "%Y-%m-%d"), dates_raw[0])
        t_end = datetime.strptime(w["end_date"], "%Y-%m-%d")
        
        # 配色定義
        if w["type"] == "上漲":
            color = "#ffcccc" # 淺紅
            label = "Upward Segment" if not legend_added["上漲"] else ""
            legend_added["上漲"] = True
        elif w["type"] == "下跌":
            color = "#ccffcc" # 淺綠
            label = "Downward Segment" if not legend_added["下跌"] else ""
            legend_added["下跌"] = True
        else: # 盤整
            color = "#ffffcc" # 淺黃
            label = "Consolidation Segment" if not legend_added["盤整"] else ""
            legend_added["盤整"] = True
            
        ax.axvspan(t_start, t_end, color=color, alpha=0.5, label=label, zorder=1)
        
    ax.set_title(f"{args.symbol} Market Segments (Upward / Downward / Consolidation)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price / Index Point", fontsize=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    ax.grid(True, linestyle="--", alpha=0.4, zorder=2)
    ax.legend(loc="upper left")
    
    plt.savefig(args.output_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"三相波段分類走勢圖已儲存至: {args.output_img}")

if __name__ == "__main__":
    main()
