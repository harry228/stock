#!/usr/bin/env python3
"""
RSI 背離分析 Web Server (V2 參數版)
參數: extreme_window=5, min_interval=5, max_dist=60
"""
import json
import os
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse

# 從原始 server 匯入共用功能
from server import (
    SYMBOL, RSI_PERIOD, LOOKBACK, FUTURE_DAYS, WEB_DIR,
    load_or_fetch, compute_rsi, compute_returns
)
from taiex_rsi_divergence import detect_divergence

# V2 參數
EXTREME_WINDOW = 5
MIN_INTERVAL = 5
MAX_DIST = 60

def get_analysis_data_v2(symbol=SYMBOL):
    records = load_or_fetch(symbol)
    closes = [r["close"] for r in records]
    rsi_values = compute_rsi(closes, RSI_PERIOD)
    
    # 執行 V2 偵測
    signals = detect_divergence(
        records, rsi_values, LOOKBACK,
        extreme_window=EXTREME_WINDOW,
        min_interval=MIN_INTERVAL,
        max_dist=MAX_DIST
    )
    
    sig_returns = compute_returns(records, signals, FUTURE_DAYS)
    for entry in sig_returns:
        idx = next(i for i, r in enumerate(records) if r["date"] == entry["date"])
        entry["rsi"] = round(rsi_values[idx], 1) if rsi_values[idx] else None
        
    return {
        "symbol": f"{symbol} (V2)",
        "data_range": f"{records[0]['date']} ~ {records[-1]['date']}",
        "total_days": len(records),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "future_days": FUTURE_DAYS,
        "signals": sig_returns,
    }

class RSIHandlerV2(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/data":
            data = get_analysis_data_v2()
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

if __name__ == "__main__":
    port = 8889
    print(f"V2 Server 啟動中: http://localhost:{port}")
    server = HTTPServer(("0.0.0.0", port), RSIHandlerV2)
    server.serve_forever()
