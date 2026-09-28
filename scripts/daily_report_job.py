import subprocess
import os
import sys
from datetime import datetime

# 路徑設定
BASE_DIR = os.path.expanduser("~/stock-analysis")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
OUTPUT_CSV = os.path.expanduser("~/storage/downloads/台指期_RSI背離訊號.csv")
SEND_EMAIL_SCRIPT = os.path.expanduser("~/send_email.py")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"執行失敗: {cmd}\n錯誤: {e.stderr}")
        return None

def main():
    print(f"[{datetime.now()}] 開始執行每日 RSI 報告作業...")
    
    # 1. 更新資料與背離分析
    print("1/3 正在更新資料與偵測背離訊號...")
    run_command(f"python3 {os.path.join(SCRIPTS_DIR, 'taiex_rsi_divergence.py')} --restore-weight")
    
    # 2. 執行 Monte Carlo 模擬並捕捉輸出
    print("2/3 正在執行 Monte Carlo 模擬分析...")
    mc_output = run_command(f"python3 {os.path.join(SCRIPTS_DIR, 'monte_carlo_bootstrap.py')} --symbols ^TWII,0050.TW")
    
    if not mc_output:
        print("模擬分析失敗，中止後續作業。")
        return

    # 強制等待檔案系統同步
    import time
    time.sleep(5)
    
    # 確保 CSV 檔案已更新為今日內容 (或上一個交易日)
    import csv
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
            if not rows:
                print("CSV 檔案為空。")
                return
            last_date_str = rows[-1][0]
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            
            # 搜尋最新的背離訊號
            last_signal_date = "無"
            for row in reversed(rows):
                if len(row) > 6 and row[6] != "0": # 假設 index 6 是訊號欄
                    last_signal_date = row[0]
                    break
    except Exception as e:
        print(f"讀取 CSV 發生錯誤: {e}")
        return

    # 判斷邏輯
    today = datetime.now().date()
    # 增加警示邏輯
    status_msg = "驗證成功" if (today - last_date).days <= 5 else "資料可能過舊"
    
    # 擷取摘要表部分
    summary_marker = "多股對比及風控決策摘要表"
    if summary_marker in mc_output:
        summary_content = mc_output.split(summary_marker)[-1]
        summary_content = f"【執行狀態】: {status_msg}\n【日期檢核】: 最新數據日期: {last_date_str}, 最新背離訊號日期: {last_signal_date}\n\n【Monte Carlo 分析摘要】\n{summary_marker}{summary_content}"
    else:
        summary_content = f"【執行狀態】: {status_msg}\n【日期檢核】: 最新數據日期: {last_date_str}, 最新背離訊號日期: {last_signal_date}\n\n無法取得模擬摘要，請檢查日誌。"

    # 3. 寄送郵件
    print("3/3 正在發送郵件報告...")
    
    # 同步資料至雲端
    print("正在同步附件至雲端...")
    run_command(f"rclone copy {OUTPUT_CSV} gdrive:stock-analysis/")
    
    subject = "RSI 每日訊號報告"
    email_cmd = [
        "python3", SEND_EMAIL_SCRIPT,
        "--subject", subject,
        "--body", summary_content,
        "--file", OUTPUT_CSV
    ]

    # 使用優化後的 run_command 執行，確保錯誤能被捕捉
    # 注意：run_command 原定義為 shell=True 字串傳入，這裡需調整為接收 list 或改為字串
    print(f"執行寄信指令...")
    try:
        subprocess.run(email_cmd, check=True)
        print("作業完成！報告已寄出。")
    except subprocess.CalledProcessError:
        print("郵件發送失敗。")

if __name__ == "__main__":
    main()
