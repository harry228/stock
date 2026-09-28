import gspread
import os
import time

# 初始化 Google Sheet 連線
# 假設 service-account.json 已存在
gc = gspread.service_account(filename=os.path.expanduser("~/stock-analysis/service-account.json"))

def create_market_sheet(sheet_name="台股類股監控_2026"):
    try:
        sh = gc.create(sheet_name)
        print(f"成功建立試算表: {sheet_name}，ID: {sh.id}")
        
        # 取得第一個分頁並寫入標題
        worksheet = sh.get_worksheet(0)
        worksheet.update('A1', [['類股名稱', '代號', '即時價格', '最後更新']])
        
        # 讀取類股清單
        sector_file = os.path.expanduser("~/stock-analysis/data/lists/twse_sectors.csv")
        with open(sector_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:] # 跳過標題
            
        data = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                name, symbol = parts[0], parts[1]
                # 轉換為 Google Finance 格式 (例如 0001.TW -> TPE:1101，這裡需注意類股對應)
                # 為簡化，我們先填入原代號，您之後可統一調整為 TPE:xxxx
                formula = f'=GOOGLEFINANCE("{symbol.replace(".TW", "")}", "price")'
                data.append([name, symbol, formula, '=NOW()'])
        
        if data:
            worksheet.update('A2', data)
            print(f"成功寫入 {len(data)} 個類股數據結構。")
        
        return sh.url
    except Exception as e:
        print(f"建立試算表失敗: {e}")
        return None

if __name__ == "__main__":
    url = create_market_sheet()
    if url:
        print(f"請透過以下連結存取您的監控表: {url}")
