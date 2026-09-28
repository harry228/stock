#!/usr/bin/env python3
"""
日週月 KD 多時間級別分析與支撐壓力繪圖排程報告腳本
- 執行 kd_multi_timeframe_plot.py 產生數據與圖表
- 讀取日線資料最後日期，進行「時效性驗證」(Data Stale Check)
- 抓取圖表圖片 (/data/data/com.termux/files/home/stock-analysis/output/kd_multi_timeframe.png)
- 使用 send_email.py 將分析報告 (文字表格) 與圖表 (附件) 寄給使用者
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
SEND_EMAIL_SCRIPT = os.path.expanduser("~/send_email.py")

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
    symbol = "^TWII"
    print(f"[{datetime.now()}] 開始執行日週月 KD 合併分析與支撐壓力報告作業...")

    # 1. 執行 KD 資料更新 (呼叫 kd_indicator.py 更新快取)
    print("1/4 正在更新日線數據快取...")
    update_cmd = ["python3", os.path.join(SCRIPTS_DIR, "kd_indicator.py"), "--symbol", symbol]
    try:
        subprocess.run(update_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"數據更新失敗: {e.stderr}")
        # 即使更新失敗也繼續，嘗試使用現有快取

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
        # 寄送失敗警報
        alert_cmd = [
            "python3", SEND_EMAIL_SCRIPT,
            "--subject", f"【警報】日週月 KD 分析報告執行失敗",
            "--body", f"分析過程出錯，請確認系統或 Yahoo Finance 連線狀態。\n\n詳細錯誤：\n{e.stderr}"
        ]
        subprocess.run(alert_cmd)
        return

    # 2. 數據時效性檢查 (Data Stale Check)
    last_date_str, delta_days = check_data_freshness(symbol)
    if last_date_str is None:
        # 數據有問題
        alert_body = f"資料檢查失敗：{delta_days}"
        print(alert_body)
        subprocess.run(["python3", SEND_EMAIL_SCRIPT, "--subject", f"【警報】KD 報告資料檢查失敗", "--body", alert_body])
        return
        
    print(f"最新數據日期: {last_date_str} (相差 {delta_days} 天)")
    
    # 依規定：若資料過期（超過 5 天），報告必須發送失敗警報而非盲目寄送
    if delta_days > 5:
        alert_body = (
            f"⚠️ 警告：加權指數數據已過期！\n"
            f"最新數據日期為: {last_date_str} (已落後當前日期 {delta_days} 天)\n"
            f"超過系統設定之 5 天上限，已中止報告。請手動確認 Yahoo Finance 資料源或排程網路連線。"
        )
        print(alert_body)
        subprocess.run(["python3", SEND_EMAIL_SCRIPT, "--subject", f"【過期警報】日週月 KD 數據過期", "--body", alert_body])
        return

    # 3. 準備信件內容
    # 依據規定，強制在信件內容中顯示「最新資料日期」
    email_body = (
        f"◆ 加權指數 (^TWII) 日週月 KD 交叉支撐壓力線報告 ◆\n"
        f"─────────────────────────────────────\n"
        f"最新數據日期: {last_date_str}\n"
        f"報告產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"數據時效驗證: 正常 (相差 {delta_days} 天)\n"
        f"─────────────────────────────────────\n\n"
    )
    
    # 將 stdout 的後半部 (支撐壓力表格) 擷取並加入信件
    table_start_marker = "【近期日、週、月 KD 交叉產生的支撐/壓力位 (最新 10 筆)】"
    if table_start_marker in stdout_output:
        report_table = stdout_output.split(table_start_marker)[-1].strip()
        email_body += f"{table_start_marker}\n\n{report_table}\n"
    else:
        email_body += f"分析細節如下：\n\n{stdout_output}\n"

    email_body += (
        "\n* 說明：\n"
        "1. 本報告已自動套用【還原權值】計算（以 adjclose 對折折價與高低點進行修正）。\n"
        "2. 日線支撐壓力線預設僅繪製最近 5 條以保持圖表易讀性；週線、月線為中長線強支撐壓力，故完整繪製。\n"
        "3. 附件包含完整生成的日週月 KD 及價格交叉對齊走勢圖表。\n"
    )

    # 4. 發送郵件 (夾帶生成的圖表)
    img_path = os.path.join(OUTPUT_DIR, "kd_multi_timeframe.png")
    subject = f"【日週月 KD 分析】加權指數支撐壓力動態報告 ({last_date_str})"
    
    # 同步至雲端 (加上 timeout 避免網路卡住)
    if os.path.exists(img_path):
        print("正在同步圖表附件至雲端...")
        try:
            subprocess.run(["rclone", "copy", img_path, "gdrive:stock-analysis/", "--contimeout", "10s", "--timeout", "15s"], check=True, capture_output=True, timeout=20)
        except Exception as e:
            print(f"rclone 同步雲端跳過/失敗 (不影響信件發送): {e}")

    email_cmd = [
        "python3", SEND_EMAIL_SCRIPT,
        "--subject", subject,
        "--body", email_body,
        "--file", img_path
    ]
    
    print("正在寄送報告郵件...")
    try:
        subprocess.run(email_cmd, check=True)
        print("報告寄送完成！")
    except subprocess.CalledProcessError as e:
        print(f"郵件發送失敗: {e}")

if __name__ == "__main__":
    main()
