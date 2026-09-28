#!/usr/bin/env python3
"""
日週月 KD 指標合併分析與支撐壓力線繪製腳本
- 讀取 ^TWII 歷史數據
- 計算日、週、月 KD (9, 3, 3)
- 偵測交叉訊號 (黃金交叉、死亡交叉)
- 於股價走勢圖上繪製對應之支撐 (黃金交叉) 與壓力 (死亡交叉) 線
- 支援還原權值
- 產出視覺化圖表
"""

import json
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# 設定路徑
BASE_DIR = os.path.expanduser("~/stock-analysis")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 設定中文字型 (針對 Android/Termux 環境優化)
font_path = "/system/fonts/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
else:
    # 備用方案
    plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback', 'DejaVu Sans', 'Arial']
    prop = None

plt.rcParams['axes.unicode_minus'] = False

def load_data(symbol, restore_weight=True):
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_daily.json")
    if not os.path.exists(cache_file):
        print(f"找不到快取檔案: {cache_file}")
        return []
        
    with open(cache_file, "r") as f:
        records = json.load(f)
        
    if not records:
        return []
        
    records = sorted(records, key=lambda x: x["date"])
    
    if restore_weight:
        for r in records:
            ratio = r["adjclose"] / r["close"] if r["close"] != 0 else 1.0
            r["open"] = round(r["open"] * ratio, 2)
            r["high"] = round(r["high"] * ratio, 2)
            r["low"] = round(r["low"] * ratio, 2)
            r["close"] = r["adjclose"]
            
    return records

def resample_data(records, timeframe='W'):
    if not records:
        return []
    
    resampled = []
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
            results.append({"date": records[i]["date"], "close": records[i]["close"], "k": None, "d": None})
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
            signals.append({
                "date": curr["date"],
                "type": "golden_cross",
                "k": curr["k"],
                "d": curr["d"],
                "close": curr["close"]
            })
        elif prev["k"] >= prev["d"] and curr["k"] < curr["d"]:
            signals.append({
                "date": curr["date"],
                "type": "death_cross",
                "k": curr["k"],
                "d": curr["d"],
                "close": curr["close"]
            })
            
    return signals

def map_signals_to_daily(signals, daily_dates):
    mapped = []
    daily_date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in daily_dates]
    
    for sig in signals:
        sig_dt = datetime.strptime(sig["date"], "%Y-%m-%d")
        found_idx = None
        for i, d_dt in enumerate(daily_date_objs):
            if d_dt >= sig_dt:
                found_idx = i
                break
        
        if found_idx is not None:
            mapped.append({
                "orig_date": sig["date"],
                "date": daily_dates[found_idx],
                "idx": found_idx,
                "type": sig["type"],
                "close": sig["close"]
            })
    return mapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="^TWII", help="標的，預設 ^TWII")
    parser.add_argument("--restore-weight", action="store_true", default=True, help="是否使用還原權值")
    parser.add_argument("--days", type=int, default=365, help="繪圖顯示的天數")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    symbol = args.symbol
    
    print(f"載入 {symbol} 資料...")
    daily_records = load_data(symbol, restore_weight=args.restore_weight)
    if not daily_records:
        print("資料載入失敗！")
        return

    daily_kd = compute_kd(daily_records)
    weekly_records = resample_data(daily_records, 'W')
    weekly_kd = compute_kd(weekly_records)
    monthly_records = resample_data(daily_records, 'M')
    monthly_kd = compute_kd(monthly_records)

    daily_signals = detect_signals(daily_kd)
    weekly_signals = detect_signals(weekly_kd)
    monthly_signals = detect_signals(monthly_kd)

    plot_days = args.days
    if len(daily_records) > plot_days:
        plot_records = daily_records[-plot_days:]
        plot_kd_d = daily_kd[-plot_days:]
    else:
        plot_records = daily_records
        plot_kd_d = daily_kd

    dates = [r["date"] for r in plot_records]
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    close_prices = [r["close"] for r in plot_records]

    w_k_daily = [None] * len(dates)
    w_d_daily = [None] * len(dates)
    m_k_daily = [None] * len(dates)
    m_d_daily = [None] * len(dates)

    for i, d in enumerate(dates):
        d_dt = datetime.strptime(d, "%Y-%m-%d")
        last_val = None
        for w in weekly_kd:
            w_dt = datetime.strptime(w["date"], "%Y-%m-%d")
            if w_dt <= d_dt:
                if w["k"] is not None: last_val = w
            else: break
        if last_val:
            w_k_daily[i] = last_val["k"]
            w_d_daily[i] = last_val["d"]

    for i, d in enumerate(dates):
        d_dt = datetime.strptime(d, "%Y-%m-%d")
        last_val = None
        for m in monthly_kd:
            m_dt = datetime.strptime(m["date"], "%Y-%m-%d")
            if m_dt <= d_dt:
                if m["k"] is not None: last_val = m
            else: break
        if last_val:
            m_k_daily[i] = last_val["k"]
            m_d_daily[i] = last_val["d"]

    start_date_str = dates[0]
    filtered_daily_sig = [s for s in daily_signals if s["date"] >= start_date_str]
    filtered_weekly_sig = [s for s in weekly_signals if s["date"] >= start_date_str]
    filtered_monthly_sig = [s for s in monthly_signals if s["date"] >= start_date_str]

    mapped_weekly_sig = map_signals_to_daily(filtered_weekly_sig, dates)
    mapped_monthly_sig = map_signals_to_daily(filtered_monthly_sig, dates)
    mapped_daily_sig = []
    for s in filtered_daily_sig:
        if s["date"] in dates:
            idx = dates.index(s["date"])
            mapped_daily_sig.append({"date": s["date"], "idx": idx, "type": s["type"], "close": s["close"]})

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Subplot 1
    ax_price = axes[0]
    ax_price.plot(dates_dt, close_prices, color='black', linewidth=1.5, label=f'{symbol} 收盤價')
    title_str = f"{symbol} 股價走勢 與 日/週/月 KD 支撐壓力線 (過去 {plot_days} 天)"
    if prop:
        ax_price.set_title(title_str, fontsize=14, fontweight='bold', fontproperties=prop)
        ax_price.set_ylabel("價格/指數", fontsize=12, fontproperties=prop)
    else:
        ax_price.set_title(title_str, fontsize=14, fontweight='bold')
        ax_price.set_ylabel("價格/指數", fontsize=12)
    ax_price.grid(True, linestyle=':', alpha=0.6)

    def draw_sr_lines(ax, mapped_sigs, linestyle, linewidth, limit_latest=None):
        sigs_to_draw = mapped_sigs[-limit_latest:] if limit_latest and len(mapped_sigs) > limit_latest else mapped_sigs
        for sig in sigs_to_draw:
            idx = sig["idx"]
            price = sig["close"]
            color = 'green' if sig["type"] == 'golden_cross' else 'red'
            ax.hlines(y=price, xmin=dates_dt[idx], xmax=dates_dt[-1], colors=color, linestyles=linestyle, linewidth=linewidth, alpha=0.7)
            ax.plot(dates_dt[idx], price, marker='o', color=color, markersize=5)

    draw_sr_lines(ax_price, mapped_daily_sig, 'dashed', 1.0, limit_latest=5)
    draw_sr_lines(ax_price, mapped_weekly_sig, 'dashdot', 1.5)
    draw_sr_lines(ax_price, mapped_monthly_sig, 'solid', 2.0)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='black', lw=1.5, label='收盤價'),
        Line2D([0], [0], color='green', lw=1.0, ls='dashed', label='日 KD 支撐線 (黃金交叉)'),
        Line2D([0], [0], color='red', lw=1.0, ls='dashed', label='日 KD 壓力線 (死亡交叉)'),
        Line2D([0], [0], color='green', lw=1.5, ls='dashdot', label='週 KD 支撐線 (黃金交叉)'),
        Line2D([0], [0], color='red', lw=1.5, ls='dashdot', label='週 KD 壓力線 (死亡交叉)'),
        Line2D([0], [0], color='green', lw=2.0, ls='solid', label='月 KD 支撐線 (黃金交叉)'),
        Line2D([0], [0], color='red', lw=2.0, ls='solid', label='月 KD 壓力線 (死亡交叉)')
    ]
    ax_price.legend(handles=custom_lines, loc='best', prop=prop if prop else None)

    # Subplot 2 (Daily KD)
    ax_daily = axes[1]
    k_vals = [x["k"] for x in plot_kd_d]
    d_vals = [x["d"] for x in plot_kd_d]
    ax_daily.plot(dates_dt, k_vals, label='K (9,3,3)', color='blue', linewidth=1)
    ax_daily.plot(dates_dt, d_vals, label='D (9,3,3)', color='orange', linewidth=1)
    ax_daily.axhline(80, color='red', linestyle='--', alpha=0.3)
    ax_daily.axhline(20, color='green', linestyle='--', alpha=0.3)
    for sig in mapped_daily_sig:
        idx = sig["idx"]
        color = 'green' if sig["type"] == 'golden_cross' else 'red'
        marker = '^' if sig["type"] == 'golden_cross' else 'v'
        y_val = plot_kd_d[idx]["k"]
        if y_val: ax_daily.plot(dates_dt[idx], y_val, marker=marker, color=color, markersize=8)
    if prop:
        ax_daily.set_ylabel("日 KD", fontsize=11, fontproperties=prop)
        ax_daily.legend(loc='upper left', prop=prop)
    else:
        ax_daily.set_ylabel("日 KD", fontsize=11)
        ax_daily.legend(loc='upper left')
    ax_daily.grid(True, linestyle=':', alpha=0.5)

    # Subplot 3 (Weekly KD)
    ax_weekly = axes[2]
    ax_weekly.plot(dates_dt, w_k_daily, label='週 K', color='purple', linewidth=1.2)
    ax_weekly.plot(dates_dt, w_d_daily, label='週 D', color='brown', linewidth=1.2)
    ax_weekly.axhline(80, color='red', linestyle='--', alpha=0.3)
    ax_weekly.axhline(20, color='green', linestyle='--', alpha=0.3)
    for sig in mapped_weekly_sig:
        idx = sig["idx"]
        color = 'green' if sig["type"] == 'golden_cross' else 'red'
        marker = '^' if sig["type"] == 'golden_cross' else 'v'
        y_val = w_k_daily[idx]
        if y_val: ax_weekly.plot(dates_dt[idx], y_val, marker=marker, color=color, markersize=8)
    if prop:
        ax_weekly.set_ylabel("週 KD", fontsize=11, fontproperties=prop)
        ax_weekly.legend(loc='upper left', prop=prop)
    else:
        ax_weekly.set_ylabel("週 KD", fontsize=11)
        ax_weekly.legend(loc='upper left')
    ax_weekly.grid(True, linestyle=':', alpha=0.5)

    # Subplot 4 (Monthly KD)
    ax_monthly = axes[3]
    ax_monthly.plot(dates_dt, m_k_daily, label='月 K', color='teal', linewidth=1.5)
    ax_monthly.plot(dates_dt, m_d_daily, label='月 D', color='darkorange', linewidth=1.5)
    ax_monthly.axhline(80, color='red', linestyle='--', alpha=0.3)
    ax_monthly.axhline(20, color='green', linestyle='--', alpha=0.3)
    for sig in mapped_monthly_sig:
        idx = sig["idx"]
        color = 'green' if sig["type"] == 'golden_cross' else 'red'
        marker = '^' if sig["type"] == 'golden_cross' else 'v'
        y_val = m_k_daily[idx]
        if y_val: ax_monthly.plot(dates_dt[idx], y_val, marker=marker, color=color, markersize=8)
    if prop:
        ax_monthly.set_ylabel("月 KD", fontsize=11, fontproperties=prop)
        ax_monthly.legend(loc='upper left', prop=prop)
    else:
        ax_monthly.set_ylabel("月 KD", fontsize=11)
        ax_monthly.legend(loc='upper left')
    ax_monthly.grid(True, linestyle=':', alpha=0.5)

    ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    output_img_path = os.path.join(OUTPUT_DIR, "kd_multi_timeframe.png")
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=150)
    print(f"圖表已成功存檔至: {output_img_path}")

    print("\n【近期日、週、月 KD 交叉產生的支撐/壓力位 (最新 10 筆)】")
    all_mapped_sigs = []
    for s in mapped_daily_sig: all_mapped_sigs.append({"tf": "日線", "date": s["date"], "type": s["type"], "price": s["close"]})
    for s in mapped_weekly_sig: all_mapped_sigs.append({"tf": "週線", "date": s["date"], "type": s["type"], "price": s["close"]})
    for s in mapped_monthly_sig: all_mapped_sigs.append({"tf": "月線", "date": s["date"], "type": s["type"], "price": s["close"]})
    all_mapped_sigs = sorted(all_mapped_sigs, key=lambda x: x["date"], reverse=True)
    
    print(f"{'層級':<6} | {'交叉日期':<10} | {'訊號類型':<8} | {'對應收盤價(支撐壓力)'}")
    print("-" * 55)
    for sig in all_mapped_sigs[:10]:
        sig_name = "黃金交叉 (支撐)" if sig["type"] == "golden_cross" else "死亡交叉 (壓力)"
        print(f"{sig['tf']:<6} | {sig['date']:<10} | {sig_name:<8} | {sig['price']:.2f}")

if __name__ == "__main__":
    main()
