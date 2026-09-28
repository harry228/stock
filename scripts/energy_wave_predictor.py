#!/usr/bin/env python3
"""
Volume Energy Wave Predictor (量價能量波段預估分析)
- Treats consolidation as an "energy accumulator" (蓄能期).
- Uses rolling volume averages to calculate normalized energy.
- Predicts the duration and magnitude of subsequent major trends (especially drops).
- Performs backtesting to evaluate the accuracy of this predictive model.
"""

import os
import json
import math
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Matplotlib configuration for Termux
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="量價能量波段預估分析")
    parser.add_argument("--symbol", type=str, default="^TWII", help="股票/指數代號")
    parser.add_argument("--range-pct", type=float, default=8.0, help="盤整波動上限百分比 (預設 8%%)")
    parser.add_argument("--min-days", type=int, default=20, help="最少盤整交易日天數 (預設 20 天)")
    parser.add_argument("--output-img", type=str, default=os.path.expanduser("~/storage/downloads/volume_energy_regression.png"))
    parser.add_argument("--output-csv", type=str, default=os.path.expanduser("~/storage/downloads/energy_prediction_backtest.csv"))
    return parser.parse_args()

def calculate_zigzag(records, threshold_pct):
    threshold = threshold_pct / 100.0
    closes = [r["close"] for r in records]
    highs = [r["high"] for r in records]
    lows = [r["low"] for r in records]
    dates = [r["date"] for r in records]
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

def pearson_corr(x, y):
    n = len(x)
    if n < 2: return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)
    if den_x == 0 or den_y == 0: return 0.0
    return num / math.sqrt(den_x * den_y)

def linear_regression(x, y):
    n = len(x)
    if n < 2: return 0.0, 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den == 0: return 0.0, 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept

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
    volumes = [r.get("volume", 0) for r in records]
    n = len(records)
    
    # 1. 計算 Rolling 120-day Volume Average
    vol_ma120 = [0] * n
    for i in range(n):
        start = max(0, i - 119)
        subset = volumes[start:i+1]
        vol_ma120[i] = sum(subset) / len(subset) if len(subset) > 0 else 1.0
        
    # 2. 計算雙重趨勢狀態
    pivots_10 = calculate_zigzag(records, 10.0)
    pivots_3 = calculate_zigzag(records, 3.0)
    
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
        
    # 3% 盤整覆蓋
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
            
            if (max_val - min_val) / start_val <= (args.range_pct / 100.0):
                trading_days = segment[-1]["index"] - segment[0]["index"]
                if trading_days >= args.min_days:
                    best_j = j
            else:
                break
                
        if best_j != -1:
            consolidations.append((pivots_3[i]["index"], pivots_3[best_j]["index"]))
            i = best_j
        else:
            i += 1
            
    for start, end in consolidations:
        for idx in range(start, end + 1):
            daily_state[idx] = "盤整"
            
    # 合併為區間
    segments = []
    curr_state = daily_state[0]
    start_idx = 0
    
    for i in range(1, n):
        if daily_state[i] != curr_state or i == n - 1:
            end_idx = i
            start_val = closes[start_idx]
            end_val = closes[end_idx]
            
            # 計算累積能量 (量能比總和)
            norm_energy = 0.0
            for k in range(start_idx, end_idx + 1):
                ma = vol_ma120[k] if vol_ma120[k] > 0 else 1.0
                norm_energy += (volumes[k] / ma)
                
            segments.append({
                "type": curr_state,
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "start_val": start_val,
                "end_val": end_val,
                "pct": (end_val - start_val) / start_val * 100,
                "days": end_idx - start_idx,
                "energy": norm_energy
            })
            curr_state = daily_state[i]
            start_idx = i
            
    # 配對：盤整(蓄能) -> 下一波趨勢(釋能)
    paired_data = []
    for idx in range(len(segments) - 1):
        if segments[idx]["type"] == "盤整" and segments[idx+1]["type"] != "盤整":
            c_seg = segments[idx]
            t_seg = segments[idx+1]
            paired_data.append({
                "c_start": c_seg["start_date"],
                "c_end": c_seg["end_date"],
                "c_energy": round(c_seg["energy"], 2),
                "c_days": c_seg["days"],
                "next_type": t_seg["type"],
                "t_start": t_seg["start_date"],
                "t_end": t_seg["end_date"],
                "t_pct": round(abs(t_seg["pct"]), 2),
                "t_days": t_seg["days"]
            })
            
    # 輸出 CSV
    import csv
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "c_start", "c_end", "c_energy", "c_days", "next_type", "t_start", "t_end", "t_pct", "t_days"
        ])
        writer.writeheader()
        writer.writerows(paired_data)
        
    print(f"\n配對歷史數據已儲存至: {args.output_csv}")
    
    # 統計能量預估的準確性 (相關係數)
    up_pairs = [p for p in paired_data if p["next_type"] == "上漲"]
    down_pairs = [p for p in paired_data if p["next_type"] == "下跌"]
    
    print("\n" + "="*80)
    print(f"             【{args.symbol}】量價能量與趨勢波段之預估準確度評估")
    print("="*80)
    
    def process_trend_analysis(pairs, label):
        print(f"【{label}分析】(樣本數: {len(pairs)} 筆)")
        if len(pairs) < 2:
            print("  - 樣本數不足，無法進行相關性回歸分析。")
            print("-" * 50)
            return None, None
            
        c_energies = [p["c_energy"] for p in pairs]
        t_pcts = [p["t_pct"] for p in pairs]
        t_days = [p["t_days"] for p in pairs]
        
        # 相關係數
        r_pct = pearson_corr(c_energies, t_pcts)
        r_days = pearson_corr(c_energies, t_days)
        
        # 回歸方程式
        slope_pct, int_pct = linear_regression(c_energies, t_pcts)
        slope_days, int_days = linear_regression(c_energies, t_days)
        
        print(f"  * 蓄積能量與「波段幅幅」相關係數 (R): {r_pct:+.4f}")
        print(f"    - 預估公式: 幅度(%) = {slope_pct:.4f} * 蓄積能量 + {int_pct:.2f}%")
        print(f"  * 蓄積能量與「波段天數」相關係數 (R): {r_days:+.4f}")
        print(f"    - 預估公式: 持續天數(天) = {slope_days:.4f} * 蓄積能量 + {int_days:.1f} 天")
        print("-" * 50)
        return (c_energies, t_pcts, t_days, slope_pct, int_pct, slope_days, int_days, r_pct, r_days)

    up_results = process_trend_analysis(up_pairs, "上漲波段 (多頭釋能)")
    down_results = process_trend_analysis(down_pairs, "下跌波段 (空頭釋能 - 大跌預估)")
    
    # 繪製回歸預測驗證圖
    if up_results or down_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot Up Trend (Magnitude)
        if up_results:
            e, p, d, s_p, i_p, s_d, i_d, r_p, r_d = up_results
            axes[0, 0].scatter(e, p, color="crimson", label="Actual Waves")
            axes[0, 0].plot(e, [s_p * x + i_p for x in e], color="darkred", linestyle="--", label=f"R = {r_p:.2f}")
            axes[0, 0].set_title("Upward Wave: Energy vs Magnitude (%)", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Accumulated Energy (Norm Vol)")
            axes[0, 0].set_ylabel("Trend Magnitude (%)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Duration
            axes[0, 1].scatter(e, d, color="crimson")
            axes[0, 1].plot(e, [s_d * x + i_d for x in e], color="darkred", linestyle="--", label=f"R = {r_d:.2f}")
            axes[0, 1].set_title("Upward Wave: Energy vs Duration (Days)", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("Accumulated Energy (Norm Vol)")
            axes[0, 1].set_ylabel("Trend Duration (Days)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot Down Trend (Magnitude)
        if down_results:
            e, p, d, s_p, i_p, s_d, i_d, r_p, r_d = down_results
            axes[1, 0].scatter(e, p, color="forestgreen", label="Actual Waves")
            axes[1, 0].plot(e, [s_p * x + i_p for x in e], color="darkgreen", linestyle="--", label=f"R = {r_p:.2f}")
            axes[1, 0].set_title("Downward Wave (Drop): Energy vs Magnitude (%)", fontsize=12, fontweight="bold")
            axes[1, 0].set_xlabel("Accumulated Energy (Norm Vol)")
            axes[1, 0].set_ylabel("Trend Magnitude (%)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Duration
            axes[1, 1].scatter(e, d, color="forestgreen")
            axes[1, 1].plot(e, [s_d * x + i_d for x in e], color="darkgreen", linestyle="--", label=f"R = {r_d:.2f}")
            axes[1, 1].set_title("Downward Wave (Drop): Energy vs Duration (Days)", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Accumulated Energy (Norm Vol)")
            axes[1, 1].set_ylabel("Trend Duration (Days)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.suptitle(f"{args.symbol} Volume-Energy Prediction Accuracy & Regressions", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(args.output_img, dpi=150)
        plt.close()
        print(f"預估模型回歸驗證圖已儲存至: {args.output_img}")

if __name__ == "__main__":
    main()
