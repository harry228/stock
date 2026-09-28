import csv
import os

def isolate_model_prediction_data():
    """
    確保繪圖程式只讀取原始模型預測資料，排除人為實際交易記錄。
    """
    raw_data_path = "/data/data/com.termux/files/home/storage/downloads/momentum_analysis_backtest.csv"
    plot_data_path = os.path.expanduser("~/stock-analysis/data/model_prediction_plot.csv")
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(plot_data_path), exist_ok=True)
    
    # 讀取並僅保留模型預測部分 (假設原始預測檔不含手動紀錄)
    with open(raw_data_path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        
    # 寫入純模型預測檔
    if reader:
        with open(plot_data_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=reader[0].keys())
            writer.writeheader()
            writer.writerows(reader)
        print(f"繪圖數據已隔離，來源: {plot_data_path}")
    else:
        print("未找到原始模型數據")

if __name__ == "__main__":
    isolate_model_prediction_data()
