#!/usr/bin/env python3
"""
比較不同波峰波谷參數的 RSI 背離表現
用法: python3 compare_params.py
"""

import os
import sys

# 把 scripts 目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taiex_rsi_divergence import (
    load_cache, compute_rsi, detect_divergence, compute_future_high_low
)

# ===== 比較的參數組 =====
PARAM_SETS = [
    {
        "name": "原始參數",
        "extreme_window": 3,
        "min_interval": 3,
        "max_dist": 0,       # 0 = 無上限
    },
    {
        "name": "建議參數",
        "extreme_window": 5,
        "min_interval": 5,
        "max_dist": 60,
    },
]

FUTURE_DAYS = [1, 3, 5, 10, 20, 60]


def analyze(records, rsi_values, params):
    """分析一組參數的表現"""
    signals = detect_divergence(
        records, rsi_values, lookback=20,
        extreme_window=params["extreme_window"],
        min_interval=params["min_interval"],
        max_dist=params["max_dist"],
    )
    
    buy_count = signals.count(1)
    sell_count = signals.count(-1)
    signal_rows = [(i, s) for i, s in enumerate(signals) if s != 0]
    
    future_data = compute_future_high_low(records, signals, FUTURE_DAYS)
    
    # 計算各天期的平均報酬
    stats = {}
    for d in FUTURE_DAYS:
        buy_returns = []
        sell_returns = []
        for idx, sig in signal_rows:
            if idx not in future_data:
                continue
            entry = future_data[idx].get(d)
            if not entry or entry[0] is None:
                continue
            h, l = entry
            close_price = records[idx]["close"]
            if sig == 1:
                buy_returns.append((h - close_price) / close_price * 100)
            else:
                sell_returns.append((close_price - l) / close_price * 100)
        
        stats[d] = {
            "buy_count": len(buy_returns),
            "sell_count": len(sell_returns),
            "buy_avg": sum(buy_returns) / len(buy_returns) if buy_returns else 0,
            "sell_avg": sum(sell_returns) / len(sell_returns) if sell_returns else 0,
        }
    
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total": buy_count + sell_count,
        "stats": stats,
    }


def main():
    # 載入快取資料
    cache = load_cache()
    if not cache or not cache.get("records"):
        print("錯誤：無快取資料，請先執行 taiex_rsi_divergence.py")
        return
    
    records = cache["records"]
    closes = [r["close"] for r in records]
    rsi_values = compute_rsi(closes, 14)
    
    print("=" * 80)
    print("RSI 背離參數比較分析")
    print(f"資料期間: {records[0]['date']} ~ {records[-1]['date']}")
    print(f"總交易日: {len(records)}")
    print("=" * 80)
    
    # 跑所有參數組
    results = []
    for params in PARAM_SETS:
        result = analyze(records, rsi_values, params)
        result["name"] = params["name"]
        result["params"] = params
        results.append(result)
    
    # ===== 輸出參數比較 =====
    print("\n【參數設定】")
    print(f"{'':20} {'原始參數':>15} {'建議參數':>15}")
    print("-" * 50)
    print(f"{'extreme_window':20} {results[0]['params']['extreme_window']:>15} {results[1]['params']['extreme_window']:>15}")
    print(f"{'min_interval':20} {results[0]['params']['min_interval']:>15} {results[1]['params']['min_interval']:>15}")
    max_dist_str_0 = "無上限" if results[0]['params']['max_dist'] == 0 else str(results[0]['params']['max_dist'])
    max_dist_str_1 = "無上限" if results[1]['params']['max_dist'] == 0 else str(results[1]['params']['max_dist'])
    print(f"{'max_dist':20} {max_dist_str_0:>15} {max_dist_str_1:>15}")
    
    # ===== 輸出訊號次數 =====
    print("\n【訊號次數】")
    print(f"{'':20} {'原始參數':>15} {'建議參數':>15} {'差異':>10}")
    print("-" * 60)
    
    r0, r1 = results[0], results[1]
    diff_buy = r1['buy_count'] - r0['buy_count']
    diff_sell = r1['sell_count'] - r0['sell_count']
    diff_total = r1['total'] - r0['total']
    
    print(f"{'買進訊號':20} {r0['buy_count']:>15} {r1['buy_count']:>15} {diff_buy:>+10}")
    print(f"{'賣出訊號':20} {r0['sell_count']:>15} {r1['sell_count']:>15} {diff_sell:>+10}")
    print(f"{'總計':20} {r0['total']:>15} {r1['total']:>15} {diff_total:>+10}")
    
    # ===== 輸出各天期平均最大獲利 =====
    print("\n【平均最大獲利 %】")
    header = f"{'天期':>6} {'原始(買)':>12} {'建議(買)':>12} {'差異':>10} {'原始(賣)':>12} {'建議(賣)':>12} {'差異':>10}"
    print(header)
    print("-" * len(header))
    
    for d in FUTURE_DAYS:
        s0 = r0['stats'][d]
        s1 = r1['stats'][d]
        
        buy_diff = s1['buy_avg'] - s0['buy_avg']
        sell_diff = s1['sell_avg'] - s0['sell_avg']
        
        buy0_str = f"{s0['buy_avg']:+.2f}%({s0['buy_count']}次)" if s0['buy_count'] > 0 else "N/A"
        buy1_str = f"{s1['buy_avg']:+.2f}%({s1['buy_count']}次)" if s1['buy_count'] > 0 else "N/A"
        sell0_str = f"{s0['sell_avg']:+.2f}%({s0['sell_count']}次)" if s0['sell_count'] > 0 else "N/A"
        sell1_str = f"{s1['sell_avg']:+.2f}%({s1['sell_count']}次)" if s1['sell_count'] > 0 else "N/A"
        
        buy_diff_str = f"{buy_diff:+.2f}%" if s0['buy_count'] > 0 and s1['buy_count'] > 0 else "N/A"
        sell_diff_str = f"{sell_diff:+.2f}%" if s0['sell_count'] > 0 and s1['sell_count'] > 0 else "N/A"
        
        print(f"{d:>5}日 {buy0_str:>12} {buy1_str:>12} {buy_diff_str:>10} {sell0_str:>12} {sell1_str:>12} {sell_diff_str:>10}")
    
    # ===== 結論 =====
    print("\n【結論】")
    
    # 比較 20 日和 60 日的表現
    for d in [20, 60]:
        s0_buy = r0['stats'][d]['buy_avg']
        s1_buy = r1['stats'][d]['buy_avg']
        s0_sell = r0['stats'][d]['sell_avg']
        s1_sell = r1['stats'][d]['sell_avg']
        
        print(f"\n{d} 日:")
        if s1_buy > s0_buy and r1['buy_count'] > 0:
            print(f"  買進: 建議參數較優 (平均獲利 {s1_buy:.2f}% > {s0_buy:.2f}%)")
        elif s0_buy > s1_buy and r0['buy_count'] > 0:
            print(f"  買進: 原始參數較優 (平均獲利 {s0_buy:.2f}% > {s1_buy:.2f}%)")
        else:
            print(f"  買進: 兩者相近")
            
        if s1_sell > s0_sell and r1['sell_count'] > 0:
            print(f"  賣出: 建議參數較優 (平均獲利 {s1_sell:.2f}% > {s0_sell:.2f}%)")
        elif s0_sell > s1_sell and r0['sell_count'] > 0:
            print(f"  賣出: 原始參數較優 (平均獲利 {s0_sell:.2f}% > {s1_sell:.2f}%)")
        else:
            print(f"  賣出: 兩者相近")


if __name__ == "__main__":
    main()
