import subprocess
import os
from datetime import datetime

# 路徑設定
BASE_DIR = os.path.expanduser("~/stock-analysis")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
# 定義三個報告的輸出與腳本
REPORTS = [
    {
        "name": "台股動能趨勢預估報告",
        "script": "dynamic_momentum_model.py",
        "csv": os.path.expanduser("~/storage/downloads/momentum_analysis_backtest.csv")
    }
]
SEND_EMAIL_SCRIPT = os.path.expanduser("~/send_email.py")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"執行失敗: {cmd}\n錯誤: {e.stderr}")
        return None

def main():
    print(f"[{datetime.now()}] 開始執行補寄作業...")
    
    for report in REPORTS:
        print(f"執行 {report['name']}...")
        output = run_command(f"python3 {os.path.join(SCRIPTS_DIR, report['script'])} --update --restore-weight")
        
        # 寄送郵件
        print(f"正在發送 {report['name']} 郵件...")
        subject = f"補寄: {report['name']}"
        
        # 簡單摘要擷取 (取最後 30 行)
        lines = output.splitlines() if output else ["無輸出內容"]
        summary = "\n".join(lines[-30:])
        
        email_cmd = [
            "python3", SEND_EMAIL_SCRIPT,
            "--subject", subject,
            "--body", f"此為補寄報告。\n\n【最新數據狀態】: {datetime.now().strftime('%Y-%m-%d')}\n\n{summary}",
            "--file", report["csv"]
        ]
        
        try:
            subprocess.run(email_cmd, check=True)
            print(f"{report['name']} 補寄成功。")
        except subprocess.CalledProcessError:
            print(f"{report['name']} 補寄失敗。")

if __name__ == "__main__":
    main()
