import os
import json
import sys
import time
import logging
from datetime import datetime

sys.path.append(os.path.expanduser("~/stock-analysis/scripts"))
from finmind_connector import fetch_finmind_data

BASE_DIR = os.path.expanduser("~/stock-analysis")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PROGRESS_FILE = os.path.join(DATA_DIR, "batch_progress.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "finmind_batch.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_symbols": []}

def save_progress(completed_list):
    data = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "completed_symbols": completed_list
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_latest_local_date(symbol):
    out_path = os.path.join(DATA_DIR, f"{symbol}_finmind.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if records and isinstance(records, list):
                    return records[-1].get("date")
        except Exception:
            pass
    return None

def batch_fetch(symbols=None, batch_size=3, force_full=False):
    # 讀取外部清單
    target_file = os.path.join(DATA_DIR, "lists/target_list.txt")
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            all_tasks = [line.strip() for line in f if line.strip()]
    else:
        all_tasks = ["2330", "2317", "2454", "2603", "2308", "0050", "2382", "2881", "2882", "2891"]
        
    progress = load_progress()
    completed = progress.get("completed_symbols", [])
    
    # 如果是每日更新，重置進度以確保每一檔都能檢查增量
    remaining = [t for t in all_tasks if t not in completed]
    
    if not remaining:
        print("所有批次標的已完成增量更新，重置進度循環。")
        logging.info("所有批次標的已完成增量更新，重置進度循環。")
        completed = []
        remaining = all_tasks
        save_progress(completed)
        
    batch = remaining[:batch_size]
    print(f"【FinMind 增量分批更新】本批次清單: {batch} (剩餘: {len(remaining)})")
    logging.info(f"本批次執行清單: {batch}")
    
    max_attempts = 4
    delays = [5, 15, 45, 120]
    
    for symbol in batch:
        success = False
        start_date = "2026-01-01"
        
        if not force_full:
            latest_date = get_latest_local_date(symbol)
            if latest_date:
                start_date = latest_date
                print(f"標的 {symbol} 發現本地快取，最新日期為 {latest_date}，進行增量更新...")
            else:
                print(f"標的 {symbol} 無本地快取，抓取完整歷史資料...")
        
        for attempt in range(max_attempts):
            try:
                print(f"正在更新 {symbol} (嘗試 {attempt+1}/{max_attempts}, 起始日: {start_date})...")
                new_records = fetch_finmind_data(symbol, start_date=start_date)
                
                out_path = os.path.join(DATA_DIR, f"{symbol}_finmind.json")
                existing_records = []
                if os.path.exists(out_path) and not force_full:
                    try:
                        with open(out_path, "r", encoding="utf-8") as f:
                            existing_records = json.load(f)
                    except Exception:
                        pass
                
                # 合併與去重
                if existing_records and not force_full:
                    date_map = {r["date"]: r for r in existing_records}
                    for r in new_records:
                        date_map[r["date"]] = r
                    merged_records = sorted(list(date_map.values()), key=lambda x: x["date"])
                else:
                    merged_records = new_records
                
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(merged_records, out_f, ensure_ascii=False, indent=2)
                    
                print(f"成功更新 {symbol}，總筆數: {len(merged_records)}")
                logging.info(f"成功更新 {symbol}，總筆數: {len(merged_records)}")
                success = True
                break
            except Exception as e:
                print(f"更新 {symbol} 失敗: {e}")
                logging.error(f"更新 {symbol} 失敗 (嘗試 {attempt+1}): {e}")
            
            if attempt < max_attempts - 1:
                sleep_sec = delays[attempt]
                print(f"觸發防禦冷卻，等待 {sleep_sec} 秒...")
                time.sleep(sleep_sec)
                
        if success:
            completed.append(symbol)
            save_progress(completed)
            time.sleep(3) # 請求間隔緩衝防 429
        else:
            print(f"標的 {symbol} 更新失敗，暫停本批次。")
            logging.error(f"標的 {symbol} 更新失敗，暫停本批次。")
            break
            
    print(f"本批次執行完畢。累計完成: {len(completed)}/{len(all_tasks)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FinMind 增量分批更新工具")
    parser.add_argument("--full", action="store_true", help="強制重新抓取完整資料")
    args = parser.parse_args()
    batch_fetch(force_full=args.full)
