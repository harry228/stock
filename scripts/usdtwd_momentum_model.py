#!/usr/bin/env python3
"""
USD/TWD Momentum & Wave Symmetry Model (美元兌新台幣量價動能與多維對稱模型)
- Tailored specifically for currency markets:
  1. Low-volatility ZigZag thresholds (Large trend: 1.5%, Short-term: 0.6%)
  2. Currency consolidation rules (range_pct=1.0%, min_days=10)
  3. Integrates Dollar Index (DXY), Korean Won (KRW), US Treasury Yields, and Foreign Capital Flow
  4. Corrected wave projection: Uses absolute wave high/low (peak/trough) for equal-amplitude symmetry targets.
"""

import os
import csv
import json
import math
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set font for Termux / Android environment
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_CSV = os.path.expanduser("~/stock-analysis/data/usdtwd_combined_data.csv")
OUTPUT_IMG = os.path.expanduser("~/storage/downloads/usdtwd_momentum_analysis.png")
OUTPUT_CSV = os.path.expanduser("~/storage/downloads/usdtwd_momentum_backtest.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="美元兌新台幣動能與多維對稱模型分析")
    parser.add_argument("--days", type=int, default=730, help="圖表繪製歷史天數 (預設: 2年)")
    parser.add_argument("--large-pct", type=float, default=1.5, help="大波段轉折百分比門檻 (預設: 1.5%%)")
    parser.add_argument("--small-pct", type=float, default=0.6, help="短線預警轉折百分比門檻 (預設: 0.6%%)")
    parser.add_argument("--range-pct", type=float, default=1.0, help="盤整波動百分比門檻 (預設: 1.0%%)")
    parser.add_argument("--min-days", type=int, default=10, help="最少盤整天數 (預設: 10天)")
    return parser.parse_args()

def load_data():
    if not os.path.exists(DATA_CSV):
        print(f"錯誤: 找不到合併後的數據源 {DATA_CSV}，請先執行抓取程序！")
        return []
        
    records = []
    with open(DATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "date": row["date"],
                    "usdtwd": float(row["usdtwd"]),
                    "dxy": float(row["dxy"]) if row.get("dxy") else None,
                    "usdkrw": float(row["usdkrw"]) if row.get("usdkrw") else None,
                    "eurusd": float(row["eurusd"]) if row.get("eurusd") else None,
                    "usdjpy": float(row["usdjpy"]) if row.get("usdjpy") else None,
                    "us10y": float(row["us10y"]) if row.get("us10y") else None,
                    "us5y": float(row["us5y"]) if row.get("us5y") else None,
                    "foreign_net": float(row["foreign_net"]) if row.get("foreign_net") else 0.0
                })
            except (ValueError, TypeError):
                continue
    return records

def calculate_zigzag(records, threshold_pct):
    threshold = threshold_pct / 100.0
    closes = [r["usdtwd"] for r in records]
    dates = [r["date"] for r in records]
    n = len(records)
    if n == 0: return []
    
    pivots = []
    direction = 0
    extreme_val = closes[0]
    extreme_idx = 0
    pivots.append({"date": dates[0], "type": "start", "value": closes[0], "index": 0})
    
    for i in range(1, n):
        val = closes[i]
        if direction == 0:
            if val > extreme_val: extreme_val, extreme_idx = val, i
            if val <= pivots[0]["value"] * (1 - threshold): direction, extreme_val, extreme_idx = -1, val, i
            elif val >= pivots[0]["value"] * (1 + threshold): direction, extreme_val, extreme_idx = 1, val, i
        elif direction == 1:
            if val > extreme_val: extreme_val, extreme_idx = val, i
            if val <= extreme_val * (1 - threshold):
                pivots.append({"date": dates[extreme_idx], "type": "peak", "value": extreme_val, "index": extreme_idx})
                direction, extreme_val, extreme_idx = -1, val, i
        elif direction == -1:
            if val < extreme_val: extreme_val, extreme_idx = val, i
            if val >= extreme_val * (1 + threshold):
                pivots.append({"date": dates[extreme_idx], "type": "trough", "value": extreme_val, "index": extreme_idx})
                direction, extreme_val, extreme_idx = 1, val, i
                
    pivots.append({"date": dates[-1], "type": "end", "value": closes[-1], "index": n - 1})
    return pivots

def main():
    args = parse_args()
    records = load_data()
    if not records:
        print("資料為空，無法執行模型預測！")
        return
        
    n = len(records)
    dates = [r["date"] for r in records]
    closes = [r["usdtwd"] for r in records]
    
    # 1. 計算多時間級別轉折點
    pivots_large = calculate_zigzag(records, args.large_pct)
    pivots_small = calculate_zigzag(records, args.small_pct)
    
    # 2. 定義每日趨勢狀態 (上漲/下跌)
    daily_state = []
    for i in range(n):
        state = "上漲"
        for j in range(len(pivots_large) - 1):
            p1 = pivots_large[j]
            p2 = pivots_large[j+1]
            if p1["index"] <= i <= p2["index"]:
                state = "上漲" if p2["value"] > p1["value"] else "下跌"
                break
        daily_state.append(state)
        
    # 3. 識別盤整期 (低波動寬度 & 連續日數)
    consolidations = []
    i = 0
    while i < len(pivots_small) - 1:
        best_j = -1
        for j in range(i + 2, len(pivots_small)):
            segment = pivots_small[i:j+1]
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
            consolidations.append((pivots_small[i]["index"], pivots_small[best_j]["index"]))
            i = best_j
        else:
            i += 1
            
    for start, end in consolidations:
        for idx in range(start, end + 1):
            daily_state[idx] = "盤整"
            
    # 4. 生成波段區間資訊 (使用最高與最低點計算波段幅度)
    segments = []
    curr_state = daily_state[0]
    start_idx = 0
    
    for i in range(1, n):
        if daily_state[i] != curr_state or i == n - 1:
            end_idx = i
            seg_closes = closes[start_idx:end_idx+1]
            wave_high = max(seg_closes)
            wave_low = min(seg_closes)
            
            if curr_state == "下跌":
                # 下跌波段 (美元下跌 / 台幣升值)：自波段高點下跌至波段低點
                pct_change = (wave_low - wave_high) / wave_high * 100
                point_change = wave_low - wave_high
            elif curr_state == "上漲":
                # 上漲波段 (美元上漲 / 台幣貶值)：自波段低點上漲至波段高點
                pct_change = (wave_high - wave_low) / wave_low * 100
                point_change = wave_high - wave_low
            else:
                # 盤整期
                pct_change = (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100
                point_change = closes[end_idx] - closes[start_idx]
            
            # 計算外資在此波段內之累積買賣超金額
            net_flow = sum([r["foreign_net"] for r in records[start_idx:end_idx+1]])
            
            segments.append({
                "type": curr_state,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "start_val": closes[start_idx],
                "end_val": closes[end_idx],
                "wave_high": wave_high,
                "wave_low": wave_low,
                "pct": round(pct_change, 2),
                "point_change": round(point_change, 3),
                "days": end_idx - start_idx,
                "foreign_flow_bil": round(net_flow / 1e9, 2)
            })
            curr_state = daily_state[i]
            start_idx = i

    # 計算近期外資與跨市場數據
    latest_rec = records[-1]
    prev_5_recs = records[-5:] if n >= 5 else records
    prev_20_recs = records[-20:] if n >= 20 else records
    
    flow_5d = sum([r["foreign_net"] for r in prev_5_recs]) / 1e9
    flow_20d = sum([r["foreign_net"] for r in prev_20_recs]) / 1e9
    
    # 歐、日、韓、DXY近期趨勢
    dxy_change_5d = latest_rec["dxy"] - prev_5_recs[0]["dxy"] if latest_rec["dxy"] and prev_5_recs[0]["dxy"] else 0
    krw_change_5d = latest_rec["usdkrw"] - prev_5_recs[0]["usdkrw"] if latest_rec["usdkrw"] and prev_5_recs[0]["usdkrw"] else 0
    yield_change_5d = latest_rec["us10y"] - prev_5_recs[0]["us10y"] if latest_rec["us10y"] and prev_5_recs[0]["us10y"] else 0

    # 5. 輸出 CLI 分析報告
    print("\n" + "═"*80)
    print(f"       【USD/TWD】美元兌新台幣動能效率與多維度對稱模型分析報告")
    print("═"*80)
    
    print("\n【1. 跨市場與籌碼因子即時監控】")
    print(f"  * 外資近 5 日累積淨買賣超: {flow_5d:+.2f} 十億元新台幣")
    print(f"  * 外資近 20 日累積淨買賣超: {flow_20d:+.2f} 十億元新台幣")
    print(f"  * 美元指數 DXY: {latest_rec['dxy']:.2f} (近5日變動: {dxy_change_5d:+.2f})")
    print(f"  * 美元兌韓元 KRW: {latest_rec['usdkrw']:.2f} (近5日變動: {krw_change_5d:+.2f})")
    print(f"  * 美國 10 年期公債殖利率: {latest_rec['us10y']:.3f}% (近5日變動: {yield_change_5d:+.3f}%)")
    print("-" * 80)
    
    # 歷史波段分析
    up_segs = [s for s in segments if s["type"] == "上漲"]
    down_segs = [s for s in segments if s["type"] == "下跌"]
    
    def print_segs_stats(segs, label):
        pcts = [abs(s["pct"]) for s in segs]
        days = [s["days"] for s in segs]
        flows = [s["foreign_flow_bil"] for s in segs]
        avg_pct = sum(pcts) / len(pcts) if pcts else 0
        avg_days = sum(days) / len(days) if days else 0
        avg_flow = sum(flows) / len(flows) if flows else 0
        print(f"【2. 歷史波段對稱統計 ({label})】")
        print(f"  * 歷史波段數: {len(segs)} 次")
        print(f"  * 平均波動幅度: {avg_pct:.2f}% (中位數: {sorted(pcts)[len(pcts)//2]:.2f}%)" if pcts else "  * 無資料")
        print(f"  * 平均持續天數: {avg_days:.1f} 天" if days else "  * 無資料")
        print(f"  * 平均波段外資流入: {avg_flow:+.2f} 十億元" if flows else "  * 無資料")
        print("-" * 80)
        return pcts, days
        
    up_pcts, up_days = print_segs_stats(up_segs, "台幣貶值波段 (美元兌台幣上漲)")
    down_pcts, down_days = print_segs_stats(down_segs, "台幣升值波段 (美元兌台幣下跌)")
    
    # 當前波段評估與投影
    curr_seg = segments[-1]
    curr_seg_closes = closes[curr_seg["start_idx"]:curr_seg["end_idx"]+1]
    curr_wave_high = max(curr_seg_closes)
    curr_wave_low = min(curr_seg_closes)

    print("【3. 當前 USD/TWD 波段評估與等幅對稱投影】")
    print(f"  * 當前大波段型態: {curr_seg['type']}")
    print(f"  * 波段起始日期: {curr_seg['start_date']} (起始匯率: {curr_seg['start_val']:.3f})")
    print(f"  * 波段最高點 (Peak): {curr_wave_high:.3f}")
    print(f"  * 波段最低點 (Trough): {curr_wave_low:.3f}")
    print(f"  * 當前最新匯率 ({curr_seg['end_date']}): {curr_seg['end_val']:.3f}")
    print(f"  * 本波段累計變動: {curr_seg['pct']:.2f}% (共 {curr_seg['point_change']:+.3f} 元)")
    print(f"  * 目前已持續交易日: {curr_seg['days']} 天")
    print(f"  * 本波段外資累積淨流入: {curr_seg['foreign_flow_bil']:+.2f} 十億元")
    
    # 短線轉折燈號 (0.6% 門檻)
    if len(pivots_small) >= 2:
        last_p = pivots_small[-2]
        curr_p = pivots_small[-1]
        st_state = "台幣短線轉強 (匯率自高點回落)" if last_p["type"] == "peak" else "台幣短線轉弱 (匯率自低點反彈)"
        st_days = curr_p["index"] - last_p["index"]
        st_pct = (curr_p["value"] - last_p["value"]) / last_p["value"] * 100
        print(f"\n  [短線動能預警 (0.6% 領先燈號)]")
        print(f"  * 短線波段狀態: {st_state}")
        print(f"  * 短線轉折基準: {last_p['date']} (匯率基準價 {last_p['value']:.3f})")
        print(f"  * 短線累計幅度: {st_pct:+.2f}% ({st_days} 天)")

    # 預測對稱投影 (嚴格以波段最高點/最低點計算滿足點)
    if curr_seg["type"] == "下跌": # 美元兌台幣下跌 = 台幣升值
        avg_d_pct = sum(down_pcts) / len(down_pcts) if down_pcts else 2.5
        med_d_pct = sorted(down_pcts)[len(down_pcts)//2] if down_pcts else 2.5
        
        # 修正：以波段最高點 (curr_wave_high) 計算等幅下跌滿足點，並用波段最低點 (curr_wave_low) 判斷是否已在波段中達成/超越
        target_avg = curr_wave_high * (1 - avg_d_pct / 100.0)
        target_med = curr_wave_high * (1 - med_d_pct / 100.0)
        
        status_avg = "(已達成/超越)" if curr_wave_low <= target_avg else "(尚未抵達)"
        status_med = "(已達成/超越)" if curr_wave_low <= target_med else "(尚未抵達)"
        
        print("\n  >>> 【台幣升值波段對稱投影預估 (以波段最高點計算)】 <<<")
        print(f"  * 投影基準點 (本波段最高峰): {curr_wave_high:.3f}")
        print(f"  * 本波段最低點 (盤中低點): {curr_wave_low:.3f}")
        print(f"  1. 溫和型升值滿足點 (歷史平均 -{avg_d_pct:.2f}%): 【{target_avg:.3f}】 {status_avg}")
        print(f"  2. 標準型升值滿足點 (歷史中位數 -{med_d_pct:.2f}%): 【{target_med:.3f}】 {status_med}")
        print("\n  💡 操作與避險決策建議：")
        if curr_wave_low <= target_med or curr_seg["end_val"] <= target_med:
            print("     - 台幣升值已抵達或超越歷史標準滿足區間。若外資買超買能出現停滯，建議美元需求者(如進口商/美股投資人)積極在此分批換匯。")
        else:
            print("     - 當前台幣升值波段尚處於中途階段，未到超強滿足區。可靜待匯率向投影滿足區靠攏。")
            
    elif curr_seg["type"] == "上漲": # 美元兌台幣上漲 = 台幣貶值
        avg_u_pct = sum(up_pcts) / len(up_pcts) if up_pcts else 2.5
        med_u_pct = sorted(up_pcts)[len(up_pcts)//2] if up_pcts else 2.5
        
        # 修正：以波段最低點 (curr_wave_low) 計算等幅上漲滿足點
        target_avg = curr_wave_low * (1 + avg_u_pct / 100.0)
        target_med = curr_wave_low * (1 + med_u_pct / 100.0)
        
        status_avg = "(已超越)" if curr_seg["end_val"] > target_avg else "(尚未抵達)"
        status_med = "(已超越)" if curr_seg["end_val"] > target_med else "(尚未抵達)"
        
        print("\n  >>> 【台幣貶值波段對稱投影預估 (以波段最低點計算)】 <<<")
        print(f"  * 投影基準點 (本波段最低谷): {curr_wave_low:.3f}")
        print(f"  1. 溫和型貶值滿足點 (歷史平均 +{avg_u_pct:.2f}%): 【{target_avg:.3f}】 {status_avg}")
        print(f"  2. 標準型貶值滿足點 (歷史中位數 +{med_u_pct:.2f}%): 【{target_med:.3f}】 {status_med}")
        print("\n  💡 操作與避險決策建議：")
        if curr_seg["end_val"] >= target_med:
            print("     - 台幣貶值已抵達歷史標準貶值滿足區。建議出口商積極在此拋售美元，美債/美股投資人不宜在此擴大追高美元。")
        else:
            print("     - 貶值波段尚未見到滿足對稱，伴隨美台利差與美元強勢，貶值行情可能延續。")
            
    print("═"*80)
    
    # 6. 導出回測波段數據 CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, extrasaction='ignore', fieldnames=[
            "type", "start_date", "end_date", "start_val", "end_val", "wave_high", "wave_low", "pct", "point_change", "days", "foreign_flow_bil"
        ])
        writer.writeheader()
        writer.writerows(segments)
    print(f"美元台幣波段分析明細已儲存至: {OUTPUT_CSV}")
    
    # 7. 繪圖 (雙子圖: 上圖匯率波段與顏色、下圖籌碼累積量/美債利率)
    recent_records = records[-args.days:]
    dates_raw = [datetime.strptime(r["date"], "%Y-%m-%d") for r in recent_records]
    recent_closes = [r["usdtwd"] for r in recent_records]
    recent_yield = [r["us10y"] for r in recent_records]
    
    # 計算 20 日累積外資資金流作為副圖指標
    cum_flow_20d = []
    for idx in range(len(records)):
        sub_recs = records[max(0, idx-19):idx+1]
        cum_flow_20d.append(sum([r["foreign_net"] for r in sub_recs]) / 1e9)
    recent_flow_20d = cum_flow_20d[-args.days:]
    
    start_date_str = recent_records[0]["date"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Exchange rate & background zones
    ax1.plot(dates_raw, recent_closes, color="#1f77b4", label="USD/TWD Rate", linewidth=2.0, zorder=3)
    
    legend_added = {"上漲": False, "下跌": False, "盤整": False}
    for w in segments:
        if w["end_date"] < start_date_str:
            continue
        t_start = max(datetime.strptime(w["start_date"], "%Y-%m-%d"), dates_raw[0])
        t_end = datetime.strptime(w["end_date"], "%Y-%m-%d")
        
        if w["type"] == "上漲":
            color = "#ffcccc"
            label = "TWD Depreciation Zone (USD Up)" if not legend_added["上漲"] else ""
            legend_added["上漲"] = True
        elif w["type"] == "下跌":
            color = "#ccffcc"
            label = "TWD Appreciation Zone (USD Down)" if not legend_added["下跌"] else ""
            legend_added["下跌"] = True
        else:
            color = "#ffffcc"
            label = "Consolidation Zone" if not legend_added["盤整"] else ""
            legend_added["盤整"] = True
            
        ax1.axvspan(t_start, t_end, color=color, alpha=0.4, zorder=1, label=label)
        
    ax1.set_title(f"USD/TWD Momentum & Multi-Factor Symmetry Analysis ({args.days} Days)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("USD/TWD Exchange Rate", fontsize=12)
    ax1.set_xlabel("Date (Top Subplot)", fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3, zorder=2)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Subplot 2: Foreign 20D Net Flow (Green) + US 10Y Yield (Orange)
    color_flow = "#2ca02c"
    ax2.plot(dates_raw, recent_flow_20d, color=color_flow, label="Foreign 20D Cum. Net Flow (Billion NTD)", linewidth=1.8, linestyle="--", zorder=3)
    ax2.fill_between(dates_raw, recent_flow_20d, color=color_flow, alpha=0.1)
    ax2.set_ylabel("Foreign Flow (Billion NTD)", color=color_flow, fontsize=12)
    ax2.set_xlabel("Date (Bottom Subplot)", fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_flow)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    ax2_right = ax2.twinx()
    color_yield = "#ff7f0e"
    ax2_right.plot(dates_raw, recent_yield, color=color_yield, label="US 10Y Treasury Yield (%)", linewidth=1.8, zorder=3)
    ax2_right.set_ylabel("US 10Y Treasury Yield (%)", color=color_yield, fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor=color_yield)
    
    # Combine legends for Subplot 2
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    
    # Format x-axis dates clearly on bottom subplot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart successfully saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
