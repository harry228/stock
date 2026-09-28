#!/usr/bin/env python3
"""
Generate Daily Machine Learning Classification Labels for TAIEX
Using Dual-Threshold ZigZag (Major trend + Minor consolidation detection)
Plus technical indicators: Moving Averages, Bias Rates, Bollinger Bands, KD, RSI, and RSI Divergence.
Outputs a clean daily CSV dataset ready for ML training.
No external dependencies (Pure Python Standard Library).
"""

import os
import json
import csv
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="生成用於機器學習的台股每日三相分類標籤與技術指標特徵資料集")
    parser.add_argument("--symbol", type=str, default="^TWII", help="股票/指數代號 (預設 ^TWII)")
    parser.add_argument("--major-threshold", type=float, default=10.0, help="大趨勢 ZigZag 門檻百分比 (預設 10%%)")
    parser.add_argument("--minor-threshold", type=float, default=3.0, help="小波段 ZigZag 門檻百分比 (預設 3%%)")
    parser.add_argument("--range-pct", type=float, default=8.0, help="盤整波動上限百分比 (預設 8%%)")
    parser.add_argument("--min-days", type=int, default=20, help="最少盤整交易日天數 (預設 20 天)")
    parser.add_argument("--output-csv", type=str, default=os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv"),
                        help="輸出的每日標籤與指標 CSV 路徑")
    return parser.parse_args()

# ==================== 技術指標計算 ====================

def compute_sma(closes, period):
    smas = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        smas[i] = round(sum(window) / period, 4)
    return smas

def compute_bias(closes, smas):
    bias = [None] * len(closes)
    for i in range(len(closes)):
        if smas[i] is not None and smas[i] != 0:
            bias[i] = round(((closes[i] - smas[i]) / smas[i]) * 100, 4)
    return bias

def compute_bollinger_bands(closes, period=20, num_std=2):
    mb = [None] * len(closes)
    ub = [None] * len(closes)
    lb = [None] * len(closes)
    bw = [None] * len(closes)
    pb = [None] * len(closes)
    
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = variance ** 0.5
        
        mb[i] = round(mean, 4)
        ub[i] = round(mean + num_std * std, 4)
        lb[i] = round(mean - num_std * std, 4)
        
        if mb[i] != 0:
            bw[i] = round((ub[i] - lb[i]) / mb[i], 4)
        if ub[i] != lb[i]:
            pb[i] = round((closes[i] - lb[i]) / (ub[i] - lb[i]), 4)
            
    return mb, ub, lb, bw, pb

def compute_kd(records, n=9, m1=3, m2=3):
    k_vals = [None] * len(records)
    d_vals = [None] * len(records)
    k = 50.0
    d = 50.0
    
    for i in range(len(records)):
        if i < n - 1:
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
        
        k_vals[i] = round(k, 2)
        d_vals[i] = round(d, 2)
        
    return k_vals, d_vals

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_values = [None] * period

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(round(100 - 100 / (1 + rs), 2))

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0
        loss = abs(diff) if diff < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(round(100 - 100 / (1 + rs), 2))

    return rsi_values

def find_local_extremes(data, window=5):
    peaks = []
    troughs = []
    for i in range(window, len(data) - window):
        val = data[i]
        if val is None:
            continue
        is_peak = True
        is_trough = True
        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(data):
                continue
            other = data[j]
            if other is None:
                continue
            if other >= val:
                is_peak = False
            if other <= val:
                is_trough = False
        if is_peak:
            peaks.append(i)
        if is_trough:
            troughs.append(i)
    return peaks, troughs

def detect_divergence(records, rsi_values, lookback=20, extreme_window=3, min_interval=3, rsi_extreme_window=2):
    closes = [r["close"] for r in records]
    n = len(closes)
    signals = [0] * n
    
    # 找出價格和 RSI 的波峰波谷
    price_peaks, price_troughs = find_local_extremes(closes, window=extreme_window)
    
    def get_rsi_extreme(idx, mode="min"):
        start = max(0, idx - rsi_extreme_window)
        end = min(len(rsi_values) - 1, idx + rsi_extreme_window)
        vals = [v for v in rsi_values[start : end + 1] if v is not None]
        if not vals:
            return None
        return min(vals) if mode == "min" else max(vals)
        
    # 牛背離（買進）：價格低谷 vs RSI 低谷
    for i, pt_idx in enumerate(price_troughs):
        if pt_idx < lookback:
            continue
        prev_pt = None
        for j in range(i - 1, -1, -1):
            if price_troughs[j] < pt_idx - min_interval:
                prev_pt = price_troughs[j]
                break
        if prev_pt is None:
            continue
        if closes[pt_idx] >= closes[prev_pt]:
            continue
        rsi_at_curr = get_rsi_extreme(pt_idx, "min")
        rsi_at_prev = get_rsi_extreme(prev_pt, "min")
        if rsi_at_curr is None or rsi_at_prev is None:
            continue
        if rsi_at_curr > rsi_at_prev:
            signals[pt_idx] = 1
            
    # 熊背離（賣出）：價格高峰 vs RSI 高峰
    for i, pp_idx in enumerate(price_peaks):
        if pp_idx < lookback:
            continue
        prev_pp = None
        for j in range(i - 1, -1, -1):
            if price_peaks[j] < pp_idx - min_interval:
                prev_pp = price_peaks[j]
                break
        if prev_pp is None:
            continue
        if closes[pp_idx] <= closes[prev_pp]:
            continue
        rsi_at_curr = get_rsi_extreme(pp_idx, "max")
        rsi_at_prev = get_rsi_extreme(prev_pp, "max")
        if rsi_at_curr is None or rsi_at_prev is None:
            continue
        if rsi_at_curr < rsi_at_prev:
            signals[pp_idx] = -1
            
    return signals

# ==================== ZigZag 與主要邏輯 ====================

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
        print(f"錯誤: 找不到快取檔案 {cache_path}，請先執行更新腳本。")
        return
        
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        records = data["records"]
        
    n = len(records)
    if n == 0:
        print("錯誤: 快取檔案中無歷史資料。")
        return
        
    print(f"成功載入 {args.symbol} 資料，共 {n} 筆歷史交易日 ({records[0]['date']} ~ {records[-1]['date']})")
    
    # 1. 計算大趨勢與小波段 ZigZag
    pivots_major = calculate_zigzag(records, args.major_threshold)
    pivots_minor = calculate_zigzag(records, args.minor_threshold)
    
    # 2. 定義初始日趨勢 (由大趨勢決定：上漲或下跌)
    daily_state = []
    for i in range(n):
        state = "上漲"
        for j in range(len(pivots_major) - 1):
            p1 = pivots_major[j]
            p2 = pivots_major[j+1]
            if p1["index"] <= i <= p2["index"]:
                state = "上漲" if p2["value"] > p1["value"] else "下跌"
                break
        daily_state.append(state)
        
    # 3. 識別盤整期：當 minor (小波段) 轉折點在指定交易日天數內，最大與最小價格波幅在 range_pct 內
    consolidations = []
    i = 0
    while i < len(pivots_minor) - 1:
        best_j = -1
        for j in range(i + 2, len(pivots_minor)):
            segment = pivots_minor[i:j+1]
            vals = [p["value"] for p in segment]
            max_val = max(vals)
            min_val = min(vals)
            start_val = segment[0]["value"]
            
            # 判斷最大高低差幅是否小於等於指定限制
            if (max_val - min_val) / start_val <= (args.range_pct / 100.0):
                trading_days = segment[-1]["index"] - segment[0]["index"]
                if trading_days >= args.min_days:
                    best_j = j
            else:
                break
                
        if best_j != -1:
            consolidations.append((pivots_minor[i]["index"], pivots_minor[best_j]["index"]))
            i = best_j  # 跳過已被判定為盤整的區段
        else:
            i += 1
            
    # 4. 覆蓋盤整狀態
    for start, end in consolidations:
        for idx in range(start, end + 1):
            daily_state[idx] = "盤整"
            
    # 5. 計算各項技術指標
    closes = [r["close"] for r in records]
    
    print("計算 3/5/10/20/60/120 均線 (SMAs)...")
    sma3 = compute_sma(closes, 3)
    sma5 = compute_sma(closes, 5)
    sma10 = compute_sma(closes, 10)
    sma20 = compute_sma(closes, 20)
    sma60 = compute_sma(closes, 60)
    sma120 = compute_sma(closes, 120)
    
    print("計算乖離率 (BIASes)...")
    bias3 = compute_bias(closes, sma3)
    bias5 = compute_bias(closes, sma5)
    bias10 = compute_bias(closes, sma10)
    bias20 = compute_bias(closes, sma20)
    bias60 = compute_bias(closes, sma60)
    bias120 = compute_bias(closes, sma120)
    
    print("計算布林通道 (Bollinger Bands)...")
    bb_mid, bb_up, bb_low, bb_width, bb_pb = compute_bollinger_bands(closes, period=20)
    
    print("計算 KD (9, 3, 3) 指標...")
    k_vals, d_vals = compute_kd(records)
    
    print("計算 RSI (14) 指標...")
    rsi_vals = compute_rsi(closes, period=14)
    
    print("計算 RSI 背離訊號...")
    div_signals = detect_divergence(records, rsi_vals)
    
    print("計算燭體相對比例 (Open/High/Low Ratios)...")
    open_ratios = [None] * n
    high_ratios = [None] * n
    low_ratios = [None] * n
    for i in range(n):
        close_val = closes[i]
        if close_val > 0:
            open_ratios[i] = round((records[i]["open"] - close_val) / close_val, 6)
            high_ratios[i] = round((records[i]["high"] - close_val) / close_val, 6)
            low_ratios[i] = round((records[i]["low"] - close_val) / close_val, 6)

    print("計算 Volume 均量與量比 (Volume SMAs & Ratios)...")
    volumes = [r["volume"] for r in records]
    vol_sma5 = compute_sma(volumes, 5)
    vol_sma20 = compute_sma(volumes, 20)
    vol_sma60 = compute_sma(volumes, 60)
    
    vol_ratio5 = [None] * n
    vol_ratio20 = [None] * n
    vol_ratio60 = [None] * n
    for i in range(n):
        v = volumes[i]
        if vol_sma5[i] is not None and vol_sma5[i] > 0:
            vol_ratio5[i] = round(v / vol_sma5[i], 4)
        if vol_sma20[i] is not None and vol_sma20[i] > 0:
            vol_ratio20[i] = round(v / vol_sma20[i], 4)
        if vol_sma60[i] is not None and vol_sma60[i] > 0:
            vol_ratio60[i] = round(v / vol_sma60[i], 4)
            
    # 6. 生成機器學習格式的欄位
    label_map = {"上漲": 1, "盤整": 0, "下跌": -1}
    
    output_records = []
    for idx, r in enumerate(records):
        state = daily_state[idx]
        label = label_map[state]
        
        row = {
            "Date": r["date"],
            "Open": r["open"],
            "High": r["high"],
            "Low": r["low"],
            "Close": r["close"],
            "Volume": r["volume"],
            
            # K線燭體比例 (Stationary Candle Features)
            "Open_Ratio": open_ratios[idx] if open_ratios[idx] is not None else "",
            "High_Ratio": high_ratios[idx] if high_ratios[idx] is not None else "",
            "Low_Ratio": low_ratios[idx] if low_ratios[idx] is not None else "",
            
            # 量比指標 (Stationary Volume Ratios)
            "Volume_Ratio_5": vol_ratio5[idx] if vol_ratio5[idx] is not None else "",
            "Volume_Ratio_20": vol_ratio20[idx] if vol_ratio20[idx] is not None else "",
            "Volume_Ratio_60": vol_ratio60[idx] if vol_ratio60[idx] is not None else "",
            
            # 均線 (SMAs)
            "SMA_3": sma3[idx] if sma3[idx] is not None else "",
            "SMA_5": sma5[idx] if sma5[idx] is not None else "",
            "SMA_10": sma10[idx] if sma10[idx] is not None else "",
            "SMA_20": sma20[idx] if sma20[idx] is not None else "",
            "SMA_60": sma60[idx] if sma60[idx] is not None else "",
            "SMA_120": sma120[idx] if sma120[idx] is not None else "",
            
            # 乖離率 (BIAS)
            "BIAS_3": bias3[idx] if bias3[idx] is not None else "",
            "BIAS_5": bias5[idx] if bias5[idx] is not None else "",
            "BIAS_10": bias10[idx] if bias10[idx] is not None else "",
            "BIAS_20": bias20[idx] if bias20[idx] is not None else "",
            "BIAS_60": bias60[idx] if bias60[idx] is not None else "",
            "BIAS_120": bias120[idx] if bias120[idx] is not None else "",
            
            # 布林通道
            "BB_Middle": bb_mid[idx] if bb_mid[idx] is not None else "",
            "BB_Upper": bb_up[idx] if bb_up[idx] is not None else "",
            "BB_Lower": bb_low[idx] if bb_low[idx] is not None else "",
            "BB_Width": bb_width[idx] if bb_width[idx] is not None else "",
            "BB_PercentB": bb_pb[idx] if bb_pb[idx] is not None else "",
            
            # KD
            "K": k_vals[idx] if k_vals[idx] is not None else "",
            "D": d_vals[idx] if d_vals[idx] is not None else "",
            
            # RSI
            "RSI": rsi_vals[idx] if rsi_vals[idx] is not None else "",
            
            # 背離訊號
            "Divergence_Signal": div_signals[idx],
            
            "State": state,
            "Label": label
        }
        
        # 計算未來 N 天的真實報酬率
        for delay in [1, 3, 5, 10, 20]:
            future_idx = idx + delay
            if future_idx < n:
                ret = (closes[future_idx] - r["close"]) / r["close"]
                row[f"Future_Return_{delay}d"] = round(ret, 6)
            else:
                row[f"Future_Return_{delay}d"] = ""
                
        output_records.append(row)
        
    # 儲存為 CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # 定義所有的欄位
    fieldnames = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "SMA_3", "SMA_5", "SMA_10", "SMA_20", "SMA_60", "SMA_120",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal",
        "State", "Label"
    ] + [f"Future_Return_{delay}d" for delay in [1, 3, 5, 10, 20]]
                 
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_records)
        
    # 7. 計算與顯示統計數據 (供機器學習建模前分析)
    class_counts = {"上漲": 0, "盤整": 0, "下跌": 0}
    for s in daily_state:
        class_counts[s] += 1
        
    print("\n" + "="*50)
    print("         機器學習分類標籤分布 (Class Balance)")
    print("="*50)
    for state in ["上漲", "盤整", "下跌"]:
        count = class_counts[state]
        pct = (count / n) * 100
        code = label_map[state]
        print(f"  - 【{state}】 (Code: {code:2d}): {count:5d} 天 ({pct:.2f}%)")
    print("-" * 50)
    print(f" 總交易日數: {n} 天")
    print("="*50)
    
    # 狀態轉移矩陣 (Transition Matrix)
    transition_counts = {1: {1: 0, 0: 0, -1: 0}, 0: {1: 0, 0: 0, -1: 0}, -1: {1: 0, 0: 0, -1: 0}}
    for idx in range(n - 1):
        curr_label = label_map[daily_state[idx]]
        next_label = label_map[daily_state[idx + 1]]
        transition_counts[curr_label][next_label] += 1
        
    print("\n" + "="*50)
    print("       今日狀態轉移至明日之機率分布 (%)")
    print("="*50)
    print("  (橫軸為明日 Label，縱軸為今日 Label)")
    print("  Label: 1 = 上漲, 0 = 盤整, -1 = 下跌\n")
    print("  今日 \\ 明日 |     1 (上漲)   |     0 (盤整)   |    -1 (下跌)   |")
    print("  " + "-"*65)
    for curr_l in [1, 0, -1]:
        total_transitions = sum(transition_counts[curr_l].values())
        if total_transitions > 0:
            p_up = (transition_counts[curr_l][1] / total_transitions) * 100
            p_flat = (transition_counts[curr_l][0] / total_transitions) * 100
            p_down = (transition_counts[curr_l][-1] / total_transitions) * 100
            curr_str = f"{curr_l:2d} (上漲)" if curr_l == 1 else (f"{curr_l:2d} (盤整)" if curr_l == 0 else f"{curr_l:2d} (下跌)")
            print(f"  {curr_str:10s} |   {p_up:6.2f}%    |   {p_flat:6.2f}%    |   {p_down:6.2f}%    |")
    print("="*50)
    
    print(f"\n[完成] 每日分類標籤與技術指標特徵 CSV 已成功匯出至:\n{args.output_csv}\n")

if __name__ == "__main__":
    main()
