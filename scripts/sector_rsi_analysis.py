import os
import csv
import subprocess
from datetime import datetime
import time

# 設定路徑
LISTS_DIR = os.path.expanduser("~/stock-analysis/data/lists")
SECTORS_FILE = os.path.join(LISTS_DIR, "twse_sectors.csv")
RSI_SCRIPT = os.path.expanduser("~/stock-analysis/scripts/taiex_rsi_divergence.py")
OUTPUT_CSV = os.path.expanduser("~/storage/downloads/類股_RSI背離訊號匯總.csv")

def get_sectors():
    sectors = []
    with open(SECTORS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            # row[0] 是名稱, row[1] 是代號 (^TW50XX)
            # 將代號直接存入，不再強制補上 .TW
            sectors.append((row[1], row[0]))
    return sectors

def main():
    sectors = get_sectors()
    all_signals = []
    
    print(f"開始分析 {len(sectors)} 個類股的 RSI 背離訊號...")
    
    for symbol, name in sectors:
        print(f"正在分析: {name} ({symbol})")
        # 呼叫 RSI 腳本並輸出至臨時檔
        temp_out = os.path.join(LISTS_DIR, "temp_rsi.csv")
        cmd = [
            "python3", RSI_SCRIPT,
            "--symbol", symbol,
            "--output", temp_out
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            # 讀取並找最後一筆訊號
            with open(temp_out, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    # 假設訊號在第 6 欄 (索引 6)
                    # 格式: ["日期", "開盤", "最高", "最低", "收盤", "RSI14", "訊號"]
                    last_signal = last_row[6]
                    if last_signal != "0":
                        all_signals.append({
                            "類股": name,
                            "代號": symbol,
                            "最後訊號日期": last_row[0],
                            "訊號": "買進" if last_signal == "1" else "賣出"
                        })
        except Exception as e:
            # 捕獲並記錄錯誤，不讓整個迴圈停止
            print(f"分析 {symbol} 失敗: {e}")
            # 將失敗記錄寫入簡單的 log 檔
            with open(os.path.expanduser("~/stock-analysis/analysis_errors.log"), "a") as f:
                f.write(f"{datetime.now()}: 分析 {symbol} 失敗: {e}\n")
            continue

    # 寫入 CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["類股", "代號", "最後訊號日期", "訊號"])
        for s in all_signals:
            writer.writerow([s["類股"], s["代號"], s["最後訊號日期"], s["訊號"]])

    # 同步至 Google Drive
    print("同步至 Google Drive 中...")
    try:
        subprocess.run(["rclone", "copy", OUTPUT_CSV, "gdrive:stock-analysis/"], check=True)
        print("已同步至 Google Drive:", OUTPUT_CSV)
    except Exception as e:
        print("同步 Google Drive 失敗:", e)

    # 製作報告
    report_body = "【28 類股 RSI 背離訊號最新偵測結果】\n\n"
    for s in all_signals:
        report_body += f"{s['類股']}({s['代號']}): {s['訊號']} (日期: {s['最後訊號日期']})\n"
    
    # 寄出信件
    print("寄送報告中...")
    cmd = [
        "python3", os.path.expanduser("~/send_email.py"),
        "--subject", f"類股指數 RSI 背離訊號列表 ({datetime.now().strftime('%Y-%m-%d')})",
        "--body", report_body,
        "--file", OUTPUT_CSV
    ]
    subprocess.run(cmd, check=True)
    print("報告已寄出。")

if __name__ == "__main__":
    main()
