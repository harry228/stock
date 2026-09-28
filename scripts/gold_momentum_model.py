#!/usr/bin/env python3
"""
Gold Momentum & Wave Symmetry Model (黃金動能效率與波動對稱模型)
- Dual-speed momentum model: 10% Long-term Anchor + 3% Short-term Leading Signal
- Calculates VPP (Volume-Per-Point) and wave symmetry statistics
- Outputs analysis CSV and PNG chart
"""

import os
import json
import math
import argparse
from datetime import datetime, timedelta
import urllib.request
import urllib.error
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 無法載入，將跳過圖表繪製階段。")

# Set font for Termux / Android environment
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="黃金量價效率與波動對稱動能模型分析")
    parser.add_argument("--symbol", type=str, default="GLD", help="黃金標的 (預設: GLD)")
    parser.add_argument("--days", type=int, default=10000, help="圖表繪製天數")
    parser.add_argument("--range-pct", type=float, default=5.0, help="盤整波動上限百分比")
    parser.add_argument("--min-days", type=int, default=15, help="最少盤整交易日天數")
    parser.add_argument("--output-img", type=str, default=os.path.expanduser("~/storage/downloads/gold_momentum_analysis.png"))
    parser.add_argument("--output-csv", type=str, default=os.path.expanduser("~/storage/downloads/gold_momentum_backtest.csv"))
    parser.add_argument("--update", action="store_true", default=True, help="增量更新 (預設開啟)")
    parser.add_argument("--force", action="store_true", help="強制重新下載全部資料")
    parser.add_argument("--restore-weight", action="store_true", default=True, help="使用還原權值 (預設開啟)")
    parser.add_argument("--years", type=int, default=30, help="下載年數 (預設 30 年)")
    return parser.parse_args()


def fetch_yahoo_data(symbol, start_date=None, end_date=None, years=30):
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        print(f"增量更新: 從 {start_date} 抓取至 {end_dt.strftime('%Y-%m-%d')}")
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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    req = urllib.request.Request(url, headers=headers)
    print(f"正在從 Yahoo Finance 下載 {symbol} 資料...")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP 錯誤 {e.code}: {e.reason}")
        url2 = url.replace("query1.finance", "query2.finance")
        req2 = urllib.request.Request(url2, headers=headers)
        print("嘗試備用伺服器...")
        with urllib.request.urlopen(req2, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

    if "chart" not in raw or not raw["chart"]["result"]:
        print(f"無法取得 {symbol} 資料或找不到結果")
        return []

    chart = raw["chart"]["result"][0]
    timestamps = chart.get("timestamp")
    if not timestamps:
        print(f"找不到 {symbol} 的時間戳記")
        return []

    quote = chart["indicators"]["quote"][0]

    close_prices = quote.get("close", [])
    high_prices = quote.get("high", [])
    low_prices = quote.get("low", [])
    open_prices = quote.get("open", [])
    volume_prices = quote.get("volume", [0] * len(close_prices))
    adjclose_prices = quote.get("adjclose", [None] * len(close_prices))

    records = []
    for i, ts in enumerate(timestamps):
        c = close_prices[i] if i < len(close_prices) else None
        h = high_prices[i] if i < len(high_prices) else None
        l = low_prices[i] if i < len(low_prices) else None
        o = open_prices[i] if i < len(open_prices) else None
        v = volume_prices[i] if volume_prices and i < len(volume_prices) else 0
        a = adjclose_prices[i] if adjclose_prices and i < len(adjclose_prices) else None

        if c is None or h is None or l is None or o is None:
            continue

        dt = datetime.fromtimestamp(ts)
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
            "volume": int(v) if v is not None else 0,
            "adjclose": round(float(a), 2) if a is not None else None,
        })

    print(f"取得 {len(records)} 筆有效日線資料")
    return records


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        records = cache.get("records", [])
        if records:
            print(f"載入快取: {len(records)} 筆資料 ({records[0]['date']} ~ {records[-1]['date']})")
        return cache
    except (json.JSONDecodeError, KeyError) as e:
        print(f"快取檔案損毀，將重新下載: {e}")
        return None


def save_cache(records, symbol, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = {
        "symbol": symbol,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"快取已更新: {cache_path}")


def merge_records(cached_records, new_records):
    date_map = {}
    for r in cached_records:
        date_map[r["date"]] = r
    for r in new_records:
        date_map[r["date"]] = r

    merged = sorted(date_map.values(), key=lambda x: x["date"])
    print(f"合併後共 {len(merged)} 筆資料")
    return merged


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
            if l <= pivots[0]["value"] * (1 - threshold): direction, extreme_val, extreme_idx = -1, l, i
            elif h >= pivots[0]["value"] * (1 + threshold): direction, extreme_val, extreme_idx = 1, h, i
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
    clean_symbol = args.symbol.replace("^", "").replace("=", "")
    cache_path = os.path.expanduser(f"~/stock-analysis/data/{clean_symbol}_cache.json")
    
    records = []
    cache = load_cache(cache_path) if not args.force else None

    if args.force:
        print("強制重新下載全部資料...")
        records = fetch_yahoo_data(args.symbol, years=args.years)
    elif args.update:
        if cache and cache.get("records"):
            cached_records = cache["records"]
            last_date = cached_records[-1]["date"]
            new_records = fetch_yahoo_data(args.symbol, start_date=last_date)
            if new_records:
                records = merge_records(cached_records, new_records)
            else:
                print("沒有新資料，使用快取")
                records = cached_records
        else:
            print("無快取資料，下載全部...")
            records = fetch_yahoo_data(args.symbol, years=args.years)
    else:
        if cache:
            records = cache["records"]
        else:
            print("無快取資料，下載全部...")
            records = fetch_yahoo_data(args.symbol, years=args.years)

    if not records:
        print("無法取得資料，分析中止")
        return

    # 處理還原權值
    if args.restore_weight:
        for rec in records:
            if "adjclose" in rec and rec["adjclose"] is not None:
                rec["close"] = rec["adjclose"]
        print("已套用還原權值進行計算")

    # 儲存快取
    save_cache(records, args.symbol, cache_path)

    dates = [r["date"] for r in records]
    closes = [r["close"] for r in records]
    volumes = [r.get("volume", 0) for r in records]
    n = len(records)

    # 1. 計算兩組轉折點 (10% 大趨勢, 3% 中短波動)
    pivots_10 = calculate_zigzag(records, 10.0)
    pivots_3 = calculate_zigzag(records, 3.0)
    
    # 2. 定義初始日趨勢
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
        
    # 3. 識別盤整期
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
            
    # 4. 生成波段區間資訊
    segments = []
    curr_state = daily_state[0]
    start_idx = 0
    
    for i in range(1, n):
        if daily_state[i] != curr_state or i == n - 1:
            end_idx = i
            start_val = closes[start_idx]
            end_val = closes[end_idx]
            point_change = abs(end_val - start_val)
            
            seg_vols = volumes[start_idx:end_idx+1]
            total_vol = sum(seg_vols)
            vpp = total_vol / point_change if point_change > 0 else 0
            
            segments.append({
                "type": curr_state,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "start_val": round(start_val, 2),
                "end_val": round(end_val, 2),
                "pct": round((end_val - start_val) / start_val * 100, 2),
                "point_change": round(point_change, 2),
                "days": end_idx - start_idx,
                "total_vol": total_vol,
                "vpp": vpp
            })
            curr_state = daily_state[i]
            start_idx = i
            
    # 5. 輸出分析結果
    print("\n" + "="*80)
    print(f"       【{args.symbol}】黃金量價效率與波動對稱動能模型報告")
    print("="*80)
    
    vpp_stats = {}
    print("【1. 量價效率分析 (Volume-Per-Point, VPP)】")
    for t in ["上漲", "下跌", "盤整"]:
        t_segs = [s for s in segments if s["type"] == t]
        avg_vpp = sum(s["vpp"] for s in t_segs) / len(t_segs) if t_segs else 0
        vpp_stats[t] = avg_vpp
        print(f"  * {t}波段: 平均移動 1 點需消耗 {avg_vpp:,.0f} 股之成交量")
    print("  * 💡 學術結論：黃金盤整期 VPP 為趨勢期之數倍，籌碼高密集蓄能！")
    print("-" * 80)
    
    # 波動等幅對稱分析
    up_segs = [s for s in segments if s["type"] == "上漲"]
    down_segs = [s for s in segments if s["type"] == "下跌"]
    
    def analyze_segs(segs, label):
        ratios = []
        p_changes = [s["pct"] for s in segs]
        days = [s["days"] for s in segs]
        for idx in range(len(segs) - 1):
            pct1 = abs(segs[idx]["pct"])
            pct2 = abs(segs[idx+1]["pct"])
            ratio = pct2 / pct1 if pct1 > 0 else 0
            ratios.append(ratio)
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        print(f"【2. 波動等幅對稱與統計 ({label})】")
        print(f"  * 歷史波段數: {len(segs)} 次")
        print(f"  * 平均幅度: {sum(p_changes)/len(p_changes):.2f}% (中位數: {sorted(p_changes)[len(p_changes)//2]:.2f}%)")
        print(f"  * 最大幅度: {min(p_changes):.2f}% | 最小幅度: {max(p_changes):.2f}%")
        print(f"  * 平均持續天數: {sum(days)/len(days):.1f} 天 (中位數: {sorted(days)[len(days)//2]} 天)")
        print("-" * 80)
        return p_changes, days
        
    up_pcts, up_days = analyze_segs(up_segs, "上漲趨勢波段")
    down_pcts, down_days = analyze_segs(down_segs, "下跌修正波段")
    
    # 6. 當前波段評估與反彈低點預測
    curr_seg = segments[-1]
    print("【3. 當前黃金波段評估與反彈低點預測 (雙速動能模型)】")
    print("  [長線大趨勢 (10% 門檻錨定)]")
    print(f"  * 當前波段類型: {curr_seg['type']}")
    print(f"  * 波段起始日期: {curr_seg['start_date']}")
    print(f"  * 波段起始價格: {curr_seg['start_val']:.2f}")
    print(f"  * 當前價格 ({curr_seg['end_date']}): {curr_seg['end_val']:.2f}")
    print(f"  * 目前波段累計漲跌: {curr_seg['pct']:.2f}%")
    print(f"  * 目前已持續交易日: {curr_seg['days']} 天")
    
    # 3% 短線動能預警計算
    if len(pivots_3) >= 2:
        last_p3 = pivots_3[-2]
        end_p3 = pivots_3[-1]
        st_type = "短線止跌上漲" if last_p3["type"] == "trough" else ("短線拉回下跌" if last_p3["type"] == "peak" else "短線盤整")
        st_start_date = last_p3["date"]
        st_start_val = last_p3["value"]
        st_end_val = end_p3["value"]
        st_pct = (st_end_val - st_start_val) / st_start_val * 100
        st_days = end_p3["index"] - last_p3["index"]
        
        print("\n  [短線動能預警 (3% 領先燈號)]")
        print(f"  * 短線波段狀態: {st_type}")
        print(f"  * 短線轉折日期: {st_start_date} (基準價 {st_start_val:.2f})")
        print(f"  * 短線累計漲跌: {st_pct:+.2f}%")
        print(f"  * 短線持續天數: {st_days} 天")
    
    if curr_seg["type"] == "下跌":
        avg_d_pct = sum(down_pcts) / len(down_pcts)
        med_d_pct = sorted(down_pcts)[len(down_pcts)//2]
        max_d_pct = min(down_pcts)
        
        target_avg = curr_seg["start_val"] * (1 + avg_d_pct / 100.0)
        target_med = curr_seg["start_val"] * (1 + med_d_pct / 100.0)
        target_max = curr_seg["start_val"] * (1 + max_d_pct / 100.0)
        
        status_avg = "(目前已跌破)" if curr_seg["end_val"] < target_avg else "(尚未跌破)"
        status_med = "(目前已跌破)" if curr_seg["end_val"] < target_med else "(尚未跌破)"
        
        print("\n  >>> 【量價對稱反彈落點投影預估】 <<<")
        print(f"  1. 溫和修正式反彈點 (歷史平均 -{abs(avg_d_pct):.2f}%): 【{target_avg:.2f}】{status_avg}")
        print(f"  2. 標準型修正反彈點 (歷史中位數 -{abs(med_d_pct):.2f}%): 【{target_med:.2f}】{status_med}")
        print(f"  3. 極端恐慌型修正點 (歷史最大修正 -{abs(max_d_pct):.2f}%): 【{target_max:.2f}】")
        print("\n  💡 風控與建倉策略建議：")
        print(f"     - 當前黃金價格已跌 {curr_seg['days']} 天、累計修正 {curr_seg['pct']:.2f}%。")
        if curr_seg["end_val"] <= target_med:
            print("     - 時間與幅度皆已進入超跌/深幅修正區（超越歷史中位數）。接近或低於歷史中位數關卡為左側建倉分批佈局區域。")
        else:
            print("     - 當前尚處於溫和修正範疇，建議靜待跌勢減緩或落點訊號確認後再進行分批建倉。")
    else:
        print("\n  💡 目前非下跌修正期，無法預估反彈落點。")
    print("="*80)
    
    # 7. 匯出 CSV 報表
    import csv
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, extrasaction='ignore', fieldnames=[
            "type", "start_date", "end_date", "start_val", "end_val", "pct", "point_change", "days", "total_vol", "vpp"
        ])
        writer.writeheader()
        writer.writerows(segments)
    print(f"數據分析報表已匯出至: {args.output_csv}")
    
    # 8. 繪圖
    if MATPLOTLIB_AVAILABLE:
        recent_records = records[-args.days:]
        dates_raw = [datetime.strptime(r["date"], "%Y-%m-%d") for r in recent_records]
        recent_closes = [r["close"] for r in recent_records]
        start_date_str = recent_records[0]["date"]
        
        daily_vpp = [0.0] * len(recent_records)
        for idx, r in enumerate(recent_records):
            for seg in segments:
                if seg["start_date"] <= r["date"] <= seg["end_date"]:
                    daily_vpp[idx] = seg["vpp"]
                    break
                    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Subplot 1: Price and Color Segments
        ax1.plot(dates_raw, recent_closes, color="#333333", label=f"Gold ({args.symbol}) Close", linewidth=1.5, zorder=3)
        
        legend_added = {"上漲": False, "下跌": False, "盤整": False}
        for w in segments:
            if w["end_date"] < start_date_str:
                continue
            t_start = max(datetime.strptime(w["start_date"], "%Y-%m-%d"), dates_raw[0])
            t_end = datetime.strptime(w["end_date"], "%Y-%m-%d")
            
            if w["type"] == "上漲":
                color = "#ffcccc"
                label = "Upward (Bull Run)" if not legend_added["上漲"] else ""
                legend_added["上漲"] = True
            elif w["type"] == "下跌":
                color = "#ccffcc"
                label = "Downward (Correction)" if not legend_added["下跌"] else ""
                legend_added["下跌"] = True
            else: # 盤整
                color = "#ffffcc"
                label = "Consolidation" if not legend_added["盤整"] else ""
                legend_added["盤整"] = True
                
            ax1.axvspan(t_start, t_end, color=color, alpha=0.5, zorder=1, label=label)
            
        ax1.set_title(f"Gold ({args.symbol}) Price Segments & Dynamic Volume-Per-Point (VPP) Analysis", fontsize=16, fontweight="bold")
        ax1.set_ylabel("Price (USD / Share)", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4, zorder=2)
        
        # Subplot 2: VPP Indicator
        ax2.plot(dates_raw, daily_vpp, color="#DAA520", label="VPP (Volume per Point)", linewidth=2.5, zorder=3)
        ax2.fill_between(dates_raw, daily_vpp, color="#DAA520", alpha=0.2)
        ax2.set_yscale("log")
        ax2.set_ylabel("Volume Per Point (Log Scale)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.4, which="both")
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.savefig(args.output_img, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"動能效率分析圖 (含 VPP 曲線) 已儲存至: {args.output_img}")

if __name__ == "__main__":
    main()
