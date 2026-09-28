import requests
import sys

# 簡單的 NewsAPI 或 RSS 抓取邏輯
# 為了穩定性，使用一個輕量的 RSS 來源或公開 API
def fetch_ai_news():
    # 使用 NewsAPI (免費層級，需替換您的 API Key)
    # 若無 API Key，建議使用 RSS 來源
    # 這裡示範一個從公開 RSS 源抓取標題的簡單實現
    url = "https://technews.tw/category/artificial-intelligence/feed/"
    # 加入 User-Agent 模擬真實瀏覽器，避開部分防爬蟲機制
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            items = []
            for item in root.findall('.//item'):
                title = item.find('title').text
                link = item.find('link').text
                items.append(f"{title}\n{link}")
                if len(items) >= 5:
                    break
            return "\n\n".join(items)
        else:
            return "無法獲取最新 AI 新聞。"
    except Exception as e:
        return f"抓取失敗: {str(e)}"

if __name__ == "__main__":
    news = fetch_ai_news()
    print(news)
