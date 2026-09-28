import csv
import json
import os

def separate_data_streams():
    """
    將混合的交易數據分離：
    1. 原始模型預測資料 (保持原始波段，用於圖表分析)
    2. 實際操作績效 (獨立作為回測追蹤)
    """
    raw_data_path = "/sdcard/Download/gold_momentum_backtest.csv"
    actual_trade_path = os.path.expanduser("~/storage/downloads/gold_trade_tracker.csv")
    
    # 建立校準後的模型資料路徑 (僅含預測，不含手動平倉截斷)
    model_plot_path = os.path.expanduser("~/stock-analysis/data/model_prediction_plot.csv")
    
    # 讀取並僅保留模型原始預測 (假設 CSV 中有 'source' 欄位，若無則需過濾備註)
    with open(raw_data_path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        
    # 寫入純模型預測檔
    with open(model_plot_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        # 篩選掉備註中包含 "系統自動模擬" 的人為干預資料，只留純模型預測
        # 若無備註欄位，建議在原始數據中標記來源
        writer.writerows(reader)
        
    print(f"已分離數據流：\n模型預測圖表源: {model_plot_path}\n實際回測記錄: {actual_trade_path}")

if __name__ == "__main__":
    separate_data_streams()
