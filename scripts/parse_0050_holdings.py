import requests
from bs4 import BeautifulSoup
import csv
import os

def fetch_0050_holdings():
    url = "https://www.etfinfo.tw/etf/0050/holdings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # 尋找表格，通常網頁會有特定 id 或 class
        # 這是一個通用邏輯，根據 etfinfo 網頁結構，我們嘗試找出對應表格
        table = soup.find('table')
        if not table:
            print("找不到表格資料")
            return

        output_path = os.path.expanduser("~/stock-analysis/data/lists/0050_holdings_parsed.csv")
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # 寫入標題與資料行
            for row in table.find_all('tr'):
                cols = [ele.text.strip() for ele in row.find_all(['td', 'th'])]
                if cols:
                    writer.writerow(cols)
        
        print(f"成功解析 0050 成分股並儲存至: {output_path}")
        
    except Exception as e:
        print(f"爬取失敗: {e}")

if __name__ == "__main__":
    fetch_0050_holdings()
