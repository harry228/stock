import csv
import os

def update_trade_tracker(csv_path=os.path.expanduser("~/storage/downloads/gold_momentum_backtest.csv")):
    if not os.path.exists(csv_path):
        print(f"找不到檔案: {csv_path}，請先執行黃金模型分析腳本。")
        return

    output_path = os.path.expanduser("~/storage/downloads/gold_trade_tracker.csv")
    
    with open(csv_path, 'r', encoding='utf-8-sig') as fin:
        reader = csv.DictReader(fin)
        # 讀取現有欄位，確保模型更新後仍能讀取
        rows = list(reader)
        
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as fout:
        fieldnames = ['訊號起始日', '訊號結束日', '波段類型', '模型預估幅度(%)', '持續天數', '短線訊號日', '短線訊號類型', '實際買入日', '實際賣出日', '實際獲利(%)', '備註']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            writer.writerow({
                '訊號起始日': row.get('start_date', ''),
                '訊號結束日': row.get('end_date', ''),
                '波段類型': row.get('type', ''),
                '模型預估幅度(%)': row.get('pct', ''),
                '持續天數': row.get('days', ''),
                '短線訊號日': row.get('short_term_signal_date', ''),
                '短線訊號類型': row.get('short_term_type', ''),
                '實際買入日': '',
                '實際賣出日': '',
                '實際獲利(%)': '',
                '備註': ''
            })
            
    print(f"已更新交易績效統計表單: {output_path}")

if __name__ == "__main__":
    update_trade_tracker()
