import csv
import json
import os
from datetime import datetime

def get_price_by_date(date, records):
    for r in records:
        if r['date'] == date:
            return r['close']
    return None

def simulate_system_trading():
    # 讀取模型生成的波段原始 CSV (作為系統信號基準)
    backtest_csv = os.path.expanduser("~/storage/downloads/gold_momentum_backtest.csv")
    cache_path = os.path.expanduser("~/stock-analysis/data/GLD_cache.json")
    output_tracker = os.path.expanduser("~/storage/downloads/gold_trade_tracker.csv")
    
    if not os.path.exists(backtest_csv) or not os.path.exists(cache_path):
        print("缺少檔案，請確保模型已執行完成")
        return

    with open(cache_path, 'r', encoding='utf-8') as f:
        records = json.load(f)['records']

    # 模擬邏輯：將每個波段視為一次交易
    # 買入日期 = start_date, 賣出日期 = end_date (信號結束時平倉)
    rows = []
    with open(backtest_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            buy_date = row['start_date']
            sell_date = row['end_date']
            
            buy_price = get_price_by_date(buy_date, records)
            sell_price = get_price_by_date(sell_date, records)
            
            profit_pct = ''
            if buy_price and sell_price and buy_price > 0:
                # 若為下跌波段，假設為放空 (Short)
                if row['type'] == '下跌':
                    profit_pct = round(((buy_price - sell_price) / buy_price) * 100, 2)
                else:
                    profit_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
            
            rows.append({
                '訊號起始日': buy_date,
                '訊號結束日': sell_date,
                '波段類型': row['type'],
                '模型預估幅度(%)': row['pct'],
                '持續天數': row['days'],
                '實際買入日': buy_date,
                '實際買入價格': buy_price,
                '實際賣出日': sell_date,
                '實際賣出價格': sell_price,
                '實際獲利(%)': profit_pct,
                '備註': '系統自動模擬交易'
            })

    # 寫入統計表
    fieldnames = ['訊號起始日', '訊號結束日', '波段類型', '模型預估幅度(%)', '持續天數', '實際買入日', '實際買入價格', '實際賣出日', '實際賣出價格', '實際獲利(%)', '備註']
    with open(output_tracker, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"已完成系統交易模擬，並更新統計表: {output_tracker}")

if __name__ == "__main__":
    simulate_system_trading()
