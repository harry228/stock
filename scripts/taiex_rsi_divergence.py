#!/usr/bin/env python3
"""
台指期日盤 RSI 背離買賣訊號
- 資料來源：Yahoo Finance API（^TWII 台灣加權指數作為台指期代理）
- RSI 週期：14
- 背離偵測視窗：20 根 K 棒
- 訊號：買 = 1, 賣 = -1
- 輸出：CSV 檔案
- 支援 --update 增量更新（僅抓取快取後的新資料）
"""

import json
import csv
import os
import sys
import argparse
from api_helper import retry_on_429
import urllib.request
import urllib.error
from datetime import datetime, timedelta

# ===== 參數設定 =====
SYMBOL = "^TWII"           # 台灣加權指數（台指期日盤代理）
RSI_PERIOD = 14            # RSI 週期
LOOKBACK = 20              # 背離偵測視窗（K 棒數）
OUTPUT_FILE = os.path.expanduser("~/台指期_RSI背離訊號.csv")

# ===== 波峰波谷偵測參數 =====
EXTREME_WINDOW = 3         # 波峰波谷偵測左右看幾根 K 棒
MIN_INTERVAL = 3           # 兩個相鄰波峰/波谷最少間隔 K 棒數
MAX_DIVERGENCE_DIST = 0    # 背離比較最大距離（0=無上限）
RSI_EXTREME_WINDOW = 2     # 在價格極值點前後 N 天內尋找 RSI 極值
DATA_DIR = os.path.expanduser("~/stock-analysis/data")
CACHE_FILE = os.path.join(DATA_DIR, f"{SYMBOL.replace('^','')}_cache.json")


def parse_args():
    parser = argparse.ArgumentParser(description="RSI 背離買賣訊號分析")
    parser.add_argument(
        "--update", action="store_true",
        help="增量更新：載入快取資料，僅抓取新資料後合併"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重新下載全部資料（忽略快取）"
    )
    parser.add_argument(
        "--symbol", type=str, default=SYMBOL,
        help=f"Yahoo Finance ticker（預設: {SYMBOL}）"
    )
    parser.add_argument(
        "--years", type=int, default=10,
        help="下載年數（預設: 10，僅非 --update 模式使用）"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"CSV 輸出路徑（預設: {OUTPUT_FILE}）"
    )
    parser.add_argument(
        "--restore-weight", action="store_true",
        help="使用還原權值（調整收盤價為除權息後的價格）"
    )
    return parser.parse_args()


def load_cache():
    """載入快取資料"""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        records = cache.get("records", [])
        if records:
            print(f"載入快取: {len(records)} 筆資料 ({records[0]['date']} ~ {records[-1]['date']})")
        return cache
    except (json.JSONDecodeError, KeyError) as e:
        print(f"快取檔案損毀，將重新下載: {e}")
        return None


def save_cache(records, symbol):
    """儲存資料到快取"""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_cache.json")
    cache = {
        "symbol": symbol,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"快取已更新: {cache_path}")


from finmind_connector import fetch_finmind_data

def fetch_yahoo_data(symbol, start_date=None, end_date=None, years=10):
    """
    遷移至 FinMind API 獲取資料
    """
    print(f"正在從 FinMind 下載 {symbol} 資料...")
    return fetch_finmind_data(symbol, start_date=start_date or "2026-01-01")



def merge_records(cached_records, new_records):
    """合併快取與新資料，以日期為 key 去重"""
    date_map = {}
    for r in cached_records:
        date_map[r["date"]] = r
    for r in new_records:
        date_map[r["date"]] = r  # 新資料覆蓋舊的同日期資料

    merged = sorted(date_map.values(), key=lambda x: x["date"])
    print(f"合併後共 {len(merged)} 筆資料")
    return merged


def compute_rsi(closes, period=14):
    """計算 RSI 指標（Wilder 平滑法）"""
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_values = [None] * period  # 前 period 筆無法計算

    # 計算初始平均漲跌幅
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
        rsi_values.append(100 - 100 / (1 + rs))

    # Wilder 平滑法遞推
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


def find_local_extremes(data, col="close", window=5):
    """找出區域極值（波峰與波谷）"""
    peaks = []     # 波峰 index
    troughs = []   # 波谷 index

    for i in range(window, len(data) - window):
        val = data[i][col] if isinstance(data[i], dict) else data[i]
        if val is None:
            continue

        # 檢查是否為區域最高
        is_peak = True
        # 檢查是否為區域最低
        is_trough = True

        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(data):
                continue
            other = data[j][col] if isinstance(data[j], dict) else data[j]
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


def detect_divergence(records, rsi_values, lookback=20,
                      extreme_window=3, min_interval=3, max_dist=0):
    """
    偵測 RSI 背離
    - 牛背離（看漲）：價格創新低，RSI 未創新低 → 買進訊號 (1)
    - 熊背離（看跌）：價格創新高，RSI 未創新高 → 賣出訊號 (-1)
    """
    def get_rsi_extreme(idx, mode="min"):
        start = max(0, idx - RSI_EXTREME_WINDOW)
        end = min(len(rsi_values) - 1, idx + RSI_EXTREME_WINDOW)
        vals = [v for v in rsi_values[start : end + 1] if v is not None]
        if not vals:
            return None
        return min(vals) if mode == "min" else max(vals)

    closes = [r["close"] for r in records]
    n = len(closes)

    # 找出價格和 RSI 的波峰波谷
    price_peaks, price_troughs = find_local_extremes(closes, window=extreme_window)

    rsi_data = []
    for i, v in enumerate(rsi_values):
        rsi_data.append(v if v is not None else None)
    rsi_peaks, rsi_troughs = find_local_extremes(rsi_data, window=extreme_window)

    signals = [0] * n

    # --- 牛背離（買進）：價格低谷 vs RSI 低谷 ---
    for i, pt_idx in enumerate(price_troughs):
        if pt_idx < lookback:
            continue

        # 找前一個價格低谷
        prev_pt = None
        for j in range(i - 1, -1, -1):
            if price_troughs[j] < pt_idx - min_interval:  # 至少間隔 min_interval 根 K 棒
                # 檢查距離上限
                if max_dist > 0 and pt_idx - price_troughs[j] > max_dist:
                    break
                prev_pt = price_troughs[j]
                break

        if prev_pt is None:
            continue

        # 價格創新低
        if closes[pt_idx] >= closes[prev_pt]:
            continue

        # 在價格低點前後 N 天內找 RSI 最低點
        rsi_at_curr = get_rsi_extreme(pt_idx, "min")
        rsi_at_prev = get_rsi_extreme(prev_pt, "min")

        if rsi_at_curr is None or rsi_at_prev is None:
            continue

        # RSI 未創新低（RSI 高於前低）= 牛背離
        if rsi_at_curr > rsi_at_prev:
            signals[pt_idx] = 1

    # --- 熊背離（賣出）：價格高峰 vs RSI 高峰 ---
    for i, pp_idx in enumerate(price_peaks):
        if pp_idx < lookback:
            continue

        # 找前一個價格高峰
        prev_pp = None
        for j in range(i - 1, -1, -1):
            if price_peaks[j] < pp_idx - min_interval:
                # 檢查距離上限
                if max_dist > 0 and pp_idx - price_peaks[j] > max_dist:
                    break
                prev_pp = price_peaks[j]
                break

        if prev_pp is None:
            continue

        # 價格創新高
        if closes[pp_idx] <= closes[prev_pp]:
            continue

        # 在價格高點前後 N 天內找 RSI 最高點
        rsi_at_curr = get_rsi_extreme(pp_idx, "max")
        rsi_at_prev = get_rsi_extreme(prev_pp, "max")

        if rsi_at_curr is None or rsi_at_prev is None:
            continue

        # RSI 未創新高（RSI 低於前高）= 熊背離
        if rsi_at_curr < rsi_at_prev:
            signals[pp_idx] = -1

    return signals


FUTURE_DAYS = [1, 3, 5, 10, 20, 60]  # 訊號後 N 日最高/最低


def compute_future_high_low(records, signals, future_days=FUTURE_DAYS):
    """
    針對每個背離訊號，計算訊號後 N 日內的最高價與最低價。
    回傳 dict: {signal_index: {day: (high, low), ...}}
    """
    n = len(records)
    result = {}

    for i, sig in enumerate(signals):
        if sig == 0:
            continue

        entry = {}
        for d in future_days:
            end_idx = min(i + d, n - 1)
            if end_idx <= i:
                entry[d] = (None, None)
                continue
            # 從 i+1 到 end_idx（含）的最高/最低
            future_highs = [records[j]["high"] for j in range(i + 1, end_idx + 1)]
            future_lows = [records[j]["low"] for j in range(i + 1, end_idx + 1)]
            entry[d] = (max(future_highs), min(future_lows))
        result[i] = entry

    return result


def save_csv(records, rsi_values, signals, future_data, output_path):
    """儲存結果到 CSV（含訊號後 N 日最高/最低）"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 建立標題列
    header = ["日期", "開盤", "最高", "最低", "收盤", "RSI14", "訊號"]
    for d in FUTURE_DAYS:
        header.append(f"{d}日最高")
        header.append(f"{d}日最低")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, rec in enumerate(records):
            rsi = rsi_values[i] if rsi_values[i] is not None else ""
            sig = signals[i]
            row = [
                rec["date"],
                rec["open"],
                rec["high"],
                rec["low"],
                rec["close"],
                rsi,
                sig,
            ]

            # 填入未來 N 日最高/最低（僅有訊號的列才有值）
            if i in future_data:
                for d in FUTURE_DAYS:
                    h, l = future_data[i].get(d, (None, None))
                    row.append(round(h, 2) if h is not None else "")
                    row.append(round(l, 2) if l is not None else "")
            else:
                for _ in FUTURE_DAYS:
                    row.append("")
                    row.append("")

            writer.writerow(row)

    print(f"\n已儲存至: {output_path}")


def main():
    args = parse_args()
    symbol = args.symbol
    output_file = args.output

    # 更新快取檔名以匹配 symbol
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_cache.json")
    # 覆寫全域 CACHE_FILE，以確保 load/save 使用正確路徑
    global CACHE_FILE
    CACHE_FILE = cache_file

    if args.force:
        # 強制重新下載
        print("強制模式：重新下載全部資料")
        records = fetch_yahoo_data(symbol, years=args.years)
    elif args.update:
        # 增量更新：載入快取 + 抓新資料
        cache = load_cache()
        if cache and cache.get("records"):
            cached_records = cache["records"]
            last_date = cached_records[-1]["date"]
            # 從快取最後一天的下一天開始抓
            new_records = fetch_yahoo_data(symbol, start_date=last_date)
            if new_records:
                records = merge_records(cached_records, new_records)
            else:
                print("沒有新資料，使用快取")
                records = cached_records
        else:
            print("無快取資料，下載全部")
            records = fetch_yahoo_data(symbol, years=args.years)
    else:
        # 預設：下載全部
        records = fetch_yahoo_data(symbol, years=args.years)

    # 取得 API 回傳後，確保我們有拿到最後的那一筆 (通常就是昨天)
    if not records:
        print("無法取得資料，請檢查網路或 ticker")
        return

    # 檢查是否為最新日期 (為了應對 Yahoo 延遲，若資料未達到最近交易日，給予提醒)
    last_record_date = datetime.strptime(records[-1]["date"], "%Y-%m-%d").date()
    today = datetime.now().date()
    # 簡單邏輯：如果是週一，前一天資料應是週五；其他日子至少應是昨天
    if (today - last_record_date).days > 3:
         print(f"警告：資料可能未更新！最後資料日期: {last_record_date}")
    else:
         print(f"資料同步完成，最新日期: {last_record_date}")

    # 如果使用還原權值，將收盤價替換為除權息後的調整價格 (adjclose)
    if args.restore_weight:
        for rec in records:
            if "adjclose" in rec and rec["adjclose"] is not None:
                rec["close"] = rec["adjclose"]
        print("已使用還原權值（調整收盤價）進行計算")
    else:
        # 檢查是否已經是還原權值（即 adjclose 已等於 close）
        sample = records[0]
        if "adjclose" in sample and sample["adjclose"] is not None and abs(sample["adjclose"] - sample["close"]) < 1e-6:
            print("資料已為還原權值，無需額外調整")
        else:
            print("未使用還原權值，使用原始收盤價")

    # 儲存快取
    save_cache(records, symbol)

    # ===== 計算 RSI =====
    closes = [r["close"] for r in records]
    rsi_values = compute_rsi(closes, RSI_PERIOD)

    # ===== 偵測背離訊號 =====
    signals = detect_divergence(records, rsi_values, LOOKBACK,
                                EXTREME_WINDOW, MIN_INTERVAL, MAX_DIVERGENCE_DIST)

    # ===== 計算訊號後 N 日最高/最低 =====
    future_data = compute_future_high_low(records, signals)

    # ===== 統計 =====
    buy_count = signals.count(1)
    sell_count = signals.count(-1)
    print(f"\n===== RSI 背離訊號統計 =====")
    print(f"資料期間: {records[0]['date']} ~ {records[-1]['date']}")
    print(f"總交易日: {len(records)}")
    print(f"RSI 週期: {RSI_PERIOD}")
    print(f"背離視窗: {LOOKBACK} 根 K 棒")
    print(f"買進訊號(1):  {buy_count} 次")
    print(f"賣出訊號(-1): {sell_count} 次")

    # ===== 列出最近訊號（含未來 N 日高低） =====
    signal_rows = [(i, s) for i, s in enumerate(signals) if s != 0]
    show_count = min(10, len(signal_rows))

    print(f"\n===== 最近 {show_count} 筆背離訊號（含未來最高/最低） =====")
    header_fmt = f"{'日期':<12} {'收盤':>10} {'RSI':>8} {'訊號':>4}"
    day_labels = []
    for d in FUTURE_DAYS:
        day_labels.append(f"{d}日最高")
        day_labels.append(f"{d}日最低")
    for dl in day_labels:
        header_fmt += f" {dl:>12}"
    print(header_fmt)
    print("-" * len(header_fmt))

    for idx, sig in signal_rows[-show_count:]:
        rsi_str = f"{rsi_values[idx]:.1f}" if rsi_values[idx] else "N/A"
        label = "買進" if sig == 1 else "賣出"
        line = f"{records[idx]['date']:<12} {closes[idx]:>10.2f} {rsi_str:>8} {sig:>4} ({label})"
        if idx in future_data:
            for d in FUTURE_DAYS:
                h, l = future_data[idx].get(d, (None, None))
                h_str = f"{h:>12.2f}" if h is not None else f"{'N/A':>12}"
                l_str = f"{l:>12.2f}" if l is not None else f"{'N/A':>12}"
                line += f" {h_str} {l_str}"
        print(line)

    # ===== 計算買進訊號平均報酬 =====
    if buy_count > 0 or sell_count > 0:
        print(f"\n===== 訊號後平均報酬統計 =====")
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
                close_price = closes[idx]
                if sig == 1:
                    # 買進：用最高價算最大獲利，用最低價算最大回撤
                    buy_returns.append(((h - close_price) / close_price * 100,
                                        (l - close_price) / close_price * 100))
                else:
                    # 賣出：用最低價算最大獲利（放空），用最高價算最大回撤
                    sell_returns.append(((close_price - l) / close_price * 100,
                                         (close_price - h) / close_price * 100))

            parts = [f"\n  {d:>2}日:"]
            if buy_returns:
                avg_max = sum(r[0] for r in buy_returns) / len(buy_returns)
                avg_min = sum(r[1] for r in buy_returns) / len(buy_returns)
                parts.append(f"買進({len(buy_returns)}次) 最大獲利+{avg_max:.2f}% 最大回撤{avg_min:.2f}%")
            if sell_returns:
                avg_max = sum(r[0] for r in sell_returns) / len(sell_returns)
                avg_min = sum(r[1] for r in sell_returns) / len(sell_returns)
                parts.append(f"賣出({len(sell_returns)}次) 最大獲利+{avg_max:.2f}% 最大回撤{avg_min:.2f}%")
            print(" | ".join(parts))

    # ===== 儲存 CSV =====
    save_csv(records, rsi_values, signals, future_data, output_file)

    # 複製 CSV 至下載資料夾
    download_path = os.path.expanduser('~/storage/downloads/')
    os.makedirs(download_path, exist_ok=True)
    try:
        import shutil
        shutil.copy(output_file, download_path)
        print(f"已將 CSV 複製到: {os.path.join(download_path, os.path.basename(output_file))}")
    except Exception as e:
        print(f"複製 CSV 時發生錯誤: {e}")


if __name__ == "__main__":
    main()
