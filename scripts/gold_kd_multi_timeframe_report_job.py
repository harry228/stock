#!/usr/bin/env python3
"""
黃金 (GLD) 日週月 KD 多時間級別分析與支撐壓力繪圖排程報告腳本
- 執行 kd_multi_timeframe_plot.py --symbol GLD 產生數據與圖表
- 讀取日線資料最後日期，進行「時效性驗證」(Data Stale Check)
- 抓取圖表圖片並寄送郵件至 neolin909@gmail.com
- 同步至 Google Drive (gdrive:Golden/)
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# 路徑設定
BASE_DIR = os.path.expanduser("~/stock-analysis")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

# 匯入 send_email
sys.path.append("/data/data/com.termux/files/home")
from send_email import send_email

def check_data_freshness(symbol):
    """
    檢查快取資料的最後日期，若超過 5 天則視為過期
    """
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('^', '')}_daily.json")
    if not os.path.exists(cache_file):
        return None, "找不到資料快取檔"
        
    try:
        with open(cache_file, "r") as f:
            records = json.load(f)
        if not records:
            return None, "快取資料為空"
            
        last_record = sorted(records, key=lambda x: x["date"])[-1]
        last_date_str = last_record["date"]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        delta_days = (datetime.now() - last_date).days
        
        return last_date_str, delta_days
    except Exception as e:
        return None, f"讀取資料錯誤: {str(e)}"

def main():
    symbol = "GLD"
    print(f"[{datetime.now()}] 開始執行黃金 (GLD) 日週月 KD 合併分析與支撐壓力報告作業...")

    # 1. 執行 KD 資料更新 (呼叫 kd_indicator.py 更新快取)
    print("1/4 正在更新黃金日線數據快取...")
    update_cmd = ["python3", os.path.join(SCRIPTS_DIR, "kd_indicator.py"), "--symbol", symbol]
    try:
        subprocess.run(update_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"數據更新失敗: {e.stderr}")

    # 2. 執行 KD 多級別繪圖與計算，並擷取其 stdout 輸出
    print("2/4 正在計算多時間級別 KD 並繪製支撐壓力線...")
    plot_script = os.path.join(SCRIPTS_DIR, "kd_multi_timeframe_plot.py")
    cmd = ["python3", plot_script, "--symbol", symbol]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout_output = result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"執行 {plot_script} 失敗！\nError: {e.stderr}"
        print(error_msg)
        send_email(
            to_email="neolin909@gmail.com",
            subject="【警報】黃金日週月 KD 分析報告執行失敗",
            body=f"分析過程出錯，請確認系統或 Yahoo Finance 連線狀態。\n\n詳細錯誤：\n{e.stderr}"
        )
        return

    # 3. 數據時效性檢查 (Data Stale Check)
    last_date_str, delta_days = check_data_freshness(symbol)
    if last_date_str is None:
        alert_body = f"資料檢查失敗：{delta_days}"
        print(alert_body)
        send_email(to_email="neolin909@gmail.com", subject="【警報】黃金 KD 報告資料檢查失敗", body=alert_body)
        return
        
    print(f"最新數據日期: {last_date_str} (相差 {delta_days} 天)")
    
    if delta_days > 5:
        alert_body = (
            f"⚠️ 警告：黃金 (GLD) 數據已過期！\n"
            f"最新數據日期為: {last_date_str} (已落後當前日期 {delta_days} 天)\n"
            f"超過系統設定之 5 天上限，已中止報告。請手動確認 Yahoo Finance 資料源或排程網路連線。"
        )
        print(alert_body)
        send_email(to_email="neolin909@gmail.com", subject="【過期警報】黃金日週月 KD 數據過期", body=alert_body)
        return

    # 4. 準備信件內容
    email_body = (
        f"◆ 黃金 (GLD) 日週月 KD 交叉支撐壓力線報告 ◆\n"
        f"─────────────────────────────────────\n"
        f"最新資料日期: {last_date_str}\n"
        f"報告產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"數據時效驗證: 正常 (相差 {delta_days} 天)\n"
        f"─────────────────────────────────────\n\n"
    )
    
    table_start_marker = "【近期日、週、月 KD 交叉產生的支撐/壓力位 (最新 10 筆)】"
    if table_start_marker in stdout_output:
        report_table = stdout_output.split(table_start_marker)[-1].strip()
        email_body += f"{table_start_marker}\n\n{report_table}\n"
    else:
        email_body += f"分析細節如下：\n\n{stdout_output}\n"

    email_body += (
        "\n* 說明：\n"
        "1. 本報告已自動套用【還原權值】計算（以 adjclose 進行修正）。\n"
        "2. 日線支撐壓力線預設僅繪製最近 5 條以保持圖表易讀性；週線、月線為中長線強支撐壓力，故完整繪製。\n"
        "3. 附件包含完整生成的黃金日週月 KD 及價格交叉對齊走勢圖表。\n"
    )

    # 5. 處理圖表路徑並複製重新命名
    src_img_path = os.path.join(OUTPUT_DIR, "kd_multi_timeframe.png")
    dst_img_path = os.path.join(OUTPUT_DIR, "gold_kd_multi_timeframe.png")
    if os.path.exists(src_img_path):
        import shutil
        shutil.copy(src_img_path, dst_img_path)

    subject = f"【黃金日週月 KD 分析】GLD 支撐壓力動態報告 ({last_date_str})"
    
    # 同步至雲端 (gdrive:Golden/)
    if os.path.exists(dst_img_path):
        print("正在同步圖表附件至 Google Drive (Golden)...")
        try:
            subprocess.run(["rclone", "copy", dst_img_path, "gdrive:Golden/"], check=True)
        except Exception as e:
            print(f"rclone 同步失敗: {e}")

    print("正在寄送報告郵件...")
    try:
        send_email(
            to_email="neolin909@gmail.com",
            subject=subject,
            body=email_body,
            attachment_path=dst_img_path
        )
        print("報告寄送完成！")
    except Exception as e:
        print(f"郵件發送失敗: {e}")

if __name__ == "__main__":
    main()
