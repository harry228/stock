#!/usr/bin/env python3
"""
RSI 背離分析 Web Server
- 提供 /api/data 端點，回傳背離訊號 + 未來 N 日報酬率 JSON
- 提供 / 首頁，載入直方圖與勝率統計的互動式網頁
- 每次載入網頁時自動從 Yahoo Finance 抓取最新資料（動態加載）
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse
import math

# ===== 參數 =====
SYMBOL = "^TWII"
RSI_PERIOD = 14
LOOKBACK = 20
FUTURE_DAYS = [1, 3, 5, 10, 20, 60]
DEFAULT_PORT = 8888
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser("~/stock-analysis/data")
CACHE_FILE = os.path.join(DATA_DIR, f"{SYMBOL.replace('^','')}_cache.json")


# ===== 資料抓取 =====
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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    req = urllib.request.Request(url, headers=headers)
    print(f"[API] 正在下載 {symbol} ...")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        url2 = url.replace("query1.finance", "query2.finance")
        req2 = urllib.request.Request(url2, headers=headers)
        with urllib.request.urlopen(req2, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

    chart = raw["chart"]["result"][0]
    timestamps = chart["timestamp"]
    quote = chart["indicators"]["quote"][0]

    records = []
    for i, ts in enumerate(timestamps):
        c = quote["close"][i]
        h = quote["high"][i]
        l = quote["low"][i]
        o = quote["open"][i]
        if c is None or h is None or l is None or o is None:
            continue
        dt = datetime.fromtimestamp(ts)
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
        })

    print(f"[API] 取得 {len(records)} 筆資料")
    return records


def load_or_fetch(symbol, years=10):
    """載入快取或重新抓取"""
    cache_path = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_cache.json")

    # 嘗試增量更新
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            cached = cache.get("records", [])
            if cached:
                last_date = cached[-1]["date"]
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                # 如果快取是今天或昨天的，直接用
                if (datetime.now() - last_dt).days <= 1:
                    print(f"[API] 使用快取 ({len(cached)} 筆, 至 {last_date})")
                    return cached
                # 否則增量抓取
                new_records = fetch_yahoo_data(symbol, start_date=last_date)
                if new_records:
                    date_map = {r["date"]: r for r in cached}
                    for r in new_records:
                        date_map[r["date"]] = r
                    merged = sorted(date_map.values(), key=lambda x: x["date"])
                    _save_cache(merged, symbol, cache_path)
                    return merged
                return cached
        except Exception as e:
            print(f"[API] 快取讀取失敗: {e}")

    # 全新下載
    records = fetch_yahoo_data(symbol, years=years)
    if records:
        _save_cache(records, symbol, cache_path)
    return records


def _save_cache(records, symbol, cache_path):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = {
        "symbol": symbol,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"[API] 快取已更新: {cache_path}")


# ===== RSI 計算 =====
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_values = [None] * period
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0)
        losses.append(abs(diff) if diff < 0 else 0)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi_values.append(100.0 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 2))

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + (diff if diff > 0 else 0)) / period
        avg_loss = (avg_loss * (period - 1) + (abs(diff) if diff < 0 else 0)) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rsi_values.append(round(100 - 100 / (1 + avg_gain / avg_loss), 2))

    return rsi_values


# ===== 背離偵測 =====
def find_local_extremes(data, window=3):
    peaks = []
    troughs = []
    for i in range(window, len(data) - window):
        val = data[i]
        if val is None:
            continue
        is_peak = is_trough = True
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


def detect_divergence(records, rsi_values, lookback=20):
    closes = [r["close"] for r in records]
    n = len(closes)
    price_peaks, price_troughs = find_local_extremes(closes, window=3)
    rsi_peaks, rsi_troughs = find_local_extremes(rsi_values, window=3)

    signals = [0] * n

    # 牛背離（買進）
    for i, pt_idx in enumerate(price_troughs):
        if pt_idx < lookback:
            continue
        prev_pt = None
        for j in range(i - 1, -1, -1):
            if price_troughs[j] < pt_idx - 3:
                prev_pt = price_troughs[j]
                break
        if prev_pt is None:
            continue
        if closes[pt_idx] >= closes[prev_pt]:
            continue
        r_curr = rsi_values[pt_idx]
        r_prev = rsi_values[prev_pt]
        if r_curr is not None and r_prev is not None and r_curr > r_prev:
            signals[pt_idx] = 1

    # 熊背離（賣出）
    for i, pp_idx in enumerate(price_peaks):
        if pp_idx < lookback:
            continue
        prev_pp = None
        for j in range(i - 1, -1, -1):
            if price_peaks[j] < pp_idx - 3:
                prev_pp = price_peaks[j]
                break
        if prev_pp is None:
            continue
        if closes[pp_idx] <= closes[prev_pp]:
            continue
        r_curr = rsi_values[pp_idx]
        r_prev = rsi_values[prev_pp]
        if r_curr is not None and r_prev is not None and r_curr < r_prev:
            signals[pp_idx] = -1

    return signals


# ===== 計算未來報酬 =====
def compute_returns(records, signals, future_days):
    n = len(records)
    result = []
    for i, sig in enumerate(signals):
        if sig == 0:
            continue
        close_price = records[i]["close"]
        entry = {
            "date": records[i]["date"],
            "close": close_price,
            "rsi": None,  # 之後填
            "signal": sig,
            "returns": {}
        }
        for d in future_days:
            end_idx = min(i + d, n - 1)
            if end_idx <= i:
                entry["returns"][str(d)] = None
                continue
            future_highs = [records[j]["high"] for j in range(i + 1, end_idx + 1)]
            future_lows = [records[j]["low"] for j in range(i + 1, end_idx + 1)]
            f_high = max(future_highs)
            f_low = min(future_lows)

            if sig == 1:  # 買進
                max_profit = (f_high - close_price) / close_price * 100
                max_drawdown = (f_low - close_price) / close_price * 100
            else:  # 賣出（放空）
                max_profit = (close_price - f_low) / close_price * 100
                max_drawdown = (close_price - f_high) / close_price * 100

            entry["returns"][str(d)] = {
                "max_profit": round(max_profit, 3),
                "max_drawdown": round(max_drawdown, 3),
            }
        result.append(entry)
    return result


# ===== 全域快取 =====
_cached_result = None


def get_analysis_data(symbol=SYMBOL, force_refresh=False):
    """取得分析資料，含快取"""
    global _cached_result
    if _cached_result is not None and not force_refresh:
        return _cached_result

    records = load_or_fetch(symbol)
    if not records:
        return {"error": "無法取得資料"}

    closes = [r["close"] for r in records]
    rsi_values = compute_rsi(closes, RSI_PERIOD)
    signals = detect_divergence(records, rsi_values, LOOKBACK)
    sig_returns = compute_returns(records, signals, FUTURE_DAYS)

    # 填入 RSI
    for entry in sig_returns:
        idx = next(i for i, r in enumerate(records) if r["date"] == entry["date"])
        entry["rsi"] = round(rsi_values[idx], 1) if rsi_values[idx] else None

    result = {
        "symbol": symbol,
        "data_range": f"{records[0]['date']} ~ {records[-1]['date']}",
        "total_days": len(records),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "future_days": FUTURE_DAYS,
        "signals": sig_returns,
    }

    _cached_result = result
    return result


# ===== HTTP Handler =====
class RSIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/data":
            self._handle_api()
        elif self.path == "/api/refresh":
            global _cached_result
            _cached_result = None
            self._handle_api()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def _handle_api(self):
        try:
            data = get_analysis_data()
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            error = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(error)

    def log_message(self, format, *args):
        print(f"[Server] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="RSI 背離分析 Web Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (預設: {DEFAULT_PORT})")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help=f"Ticker (預設: {SYMBOL})")
    parser.add_argument("--years", type=int, default=10, help="下載年數 (預設: 10)")
    args = parser.parse_args()

    symbol = args.symbol
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_cache.json")

    print(f"RSI 背離分析 Web Server")
    print(f"Ticker: {symbol}")
    print(f"Port: {args.port}")
    print(f"開啟瀏覽器: http://localhost:{args.port}")
    print(f"按 Ctrl+C 停止\n")

    # 預載資料
    print("[Server] 預載資料中...")
    get_analysis_data(symbol)
    print("[Server] 資料就緒\n")

    server = HTTPServer(("0.0.0.0", args.port), RSIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] 已停止")
        server.server_close()


if __name__ == "__main__":
    main()
