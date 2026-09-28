import requests
import os

# 將 Token 寫入腳本變數或環境，這裡直接放在 connector 中方便您調用
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiaGFycnkyMjgxOTk2QGdtYWlsLmNvbSIsImVtYWlsIjoiaGFycnkyMjgxOTk2QGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjB9.gXpfRKq844n0gLu0JwtH_z9umdDLU3nzL_J4YNm7JKc"

def fetch_finmind_data(symbol, start_date="2026-01-01"):
    # 處理代號 (去除 .TW)
    clean_symbol = str(symbol).split('.')[0]
    
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": clean_symbol,
        "start_date": start_date,
        "token": FINMIND_TOKEN
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("msg") != "success":
        raise Exception(f"FinMind API 錯誤: {data.get('msg')}")

    # 格式轉換為原本程式所需的格式
    records = []
    for row in data["data"]:
        records.append({
            "date": row["date"],
            "open": float(row["open"]),
            "high": float(row["max"]),
            "low": float(row["min"]),
            "close": float(row["close"]),
            "volume": int(row["Trading_Volume"])
        })
    return records
