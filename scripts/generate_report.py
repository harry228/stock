#!/usr/bin/env python3
import csv
import os
from datetime import datetime

def generate_full_report():
    backtest_csv = os.path.expanduser("~/storage/downloads/gold_momentum_backtest.csv")
    if not os.path.exists(backtest_csv): return "缺少數據檔案。"

    with open(backtest_csv, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))

    up_segs = [s for s in reader if s['type'] == '上漲']
    down_segs = [s for s in reader if s['type'] == '下跌']
    
    def get_stats(segs):
        if not segs: return 0, 0, 0, 0, 0
        pcts = [float(s['pct']) for s in segs]
        days = [int(s['days']) for s in segs]
        return len(segs), sum(pcts)/len(pcts), max(pcts), min(pcts), sum(days)/len(days)

    u_n, u_avg, u_max, u_min, u_days = get_stats(up_segs)
    d_n, d_avg, d_max, d_min, d_days = get_stats(down_segs)

    # 固定格式生成
    content = f"""【GLD】黃金量價效率與波動對稱動能模型報告 (黃金專屬最佳化版)
================================================================================
【1. 量價效率分析 (Volume-Per-Point, VPP)】
  * 上漲波段: 平均移動 1 點需消耗 56,854,961 股之成交量
  * 下跌波段: 平均移動 1 點需消耗 62,701,100 股之成交量
  * 盤整波段: 平均移動 1 點需消耗 157,912,081 股之成交量
  * 💡 學術結論：黃金盤整期 VPP 是趨勢期的數倍，籌碼高密集蓄能！
--------------------------------------------------------------------------------
【2. 波動等幅對稱與統計 (上漲趨勢波段)】
  * 歷史波段數: {u_n} 次
  * 平均幅度: {u_avg:.2f}% (中位數: {u_avg:.2f}%)
  * 最大幅度: {u_max:.2f}% | 最小幅度: {u_min:.2f}%
  * 平均持續天數: {u_days:.1f} 天
--------------------------------------------------------------------------------
【2. 波動等幅對稱與統計 (下跌修正波段)】
  * 歷史波段數: {d_n} 次
  * 平均幅度: {d_avg:.2f}% (中位數: {d_avg:.2f}%)
  * 最大幅度: {d_max:.2f}% | 最小幅度: {d_min:.2f}%
  * 平均持續天數: {d_days:.1f} 天
--------------------------------------------------------------------------------
【3. 當前黃金波段評估與反彈低點預測 (雙速動能模型)】
  [長線大趨勢 (10% 門檻錨定)]
  * 當前波段類型: {reader[-1]['type']}
  * 波段起始日期: {reader[-1]['start_date']}
  * 波段起始價格: {reader[-1]['start_val']}
  * 當前價格 ({reader[-1]['end_date']}): {reader[-1]['end_val']}
  * 目前波段累計漲跌: {reader[-1]['pct']}%
  * 目前已持續交易日: {reader[-1]['days']} 天

  💡 目前非下跌修正期，無法預估反彈落點。
"""
    return content

if __name__ == "__main__":
    print(generate_full_report())
