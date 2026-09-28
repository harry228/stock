import csv
import json
import os

def get_price_by_date(date, records):
    for r in records:
        if r['date'] == date:
            return r['close']
    return None

def update_gold_tracker():
    tracker_path = os.path.expanduser("~/storage/downloads/gold_trade_tracker.csv")
    cache_path = os.path.expanduser("~/stock-analysis/data/GLD_cache.json")
    
    if not os.path.exists(tracker_path) or not os.path.exists(cache_path):
        print("缺少檔案，請確保報表與快取已生成")
        return

    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
        records = cache_data['records']

    rows = []
    with open(tracker_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        # 確保有價格欄位
        if '實際買入價格' not in fieldnames:
            fieldnames.insert(6, '實際買入價格')
        if '實際賣出價格' not in fieldnames:
            fieldnames.insert(8, '實際賣出價格')
            
        for row in reader:
            # 嘗試回溯價格
            if row['實際買入日']:
                price = get_price_by_date(row['實際買入日'], records)
                row['實際買入價格'] = price if price else ''
            if row['實際賣出日']:
                price = get_price_by_date(row['實際賣出日'], records)
                row['實際賣出價格'] = price if price else ''
            rows.append(row)

    with open(tracker_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已回溯價格填入: {tracker_path}")

if __name__ == "__main__":
    update_gold_tracker()
