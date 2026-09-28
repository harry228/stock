import os
import requests
import csv

# 目標目錄
LISTS_DIR = os.path.expanduser("~/stock-analysis/data/lists")
os.makedirs(LISTS_DIR, exist_ok=True)

def fetch_twse_stocks():
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    res = requests.get(url, timeout=30)
    res.encoding = 'big5'
    with open(os.path.join(LISTS_DIR, "twse_stocks.csv"), "w", encoding="utf-8") as f:
        f.write(res.text)

def download_etf_holdings(ticker, name):
    path = os.path.join(LISTS_DIR, f"{ticker}_holdings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"請參閱 {name} 投信官方公告獲取最新成分股清單。")

def save_sectors():
    sectors = [
        ("0001", "水泥類"), ("0002", "食品類"), ("0003", "塑膠類"), ("0004", "紡織纖維類"),
        ("0005", "電機機械類"), ("0006", "電器電纜類"), ("0007", "化學工業類"), ("0008", "生技醫療類"),
        ("0009", "玻璃陶瓷類"), ("0010", "造紙類"), ("0011", "鋼鐵工業類"), ("0012", "橡膠工業類"),
        ("0013", "汽車類"), ("0014", "半導體類"), ("0015", "電腦及週邊設備類"), ("0016", "光電類"),
        ("0017", "通信網路類"), ("0018", "電子零組件類"), ("0019", "電子通路類"), ("0020", "資訊服務類"),
        ("0021", "其他電子類"), ("0022", "營建類"), ("0023", "運輸類"), ("0024", "觀光餐旅類"),
        ("0025", "金融保險類"), ("0026", "貿易百貨類"), ("0027", "油電燃氣類"), ("0028", "其他類")
    ]
    with open(os.path.join(LISTS_DIR, "twse_sectors.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["代號", "名稱"])
        writer.writerows(sectors)

def main():
    fetch_twse_stocks()
    save_sectors()
    download_etf_holdings("0050", "元大台灣50")
    download_etf_holdings("0051", "元大中型100")
    
    import subprocess
    cmd = [
        "python3", os.path.expanduser("~/send_email.py"),
        "--subject", "台股市場清單更新通知",
        "--body", "更新完成",
        "--file", os.path.join(LISTS_DIR, "twse_stocks.csv")
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
