# 台股分析專案 (Taiwan Stock Analysis)

## 專案結構

```
stock-analysis/
├── README.md           # 專案說明
├── scripts/            # 分析腳本
│   └── taiex_rsi_divergence.py   # RSI 背離訊號
├── data/               # 原始資料快取
└── output/             # 分析結果輸出
    └── 台指期_RSI背離訊號.csv
```

## 可用分析

| 分析類型 | 腳本 | 說明 |
|---------|------|------|
| RSI 背離 | `scripts/taiex_rsi_divergence.py` | 14 期 RSI，20 根 K 棒視窗，牛/熊背離偵測 |

## 使用方式

```bash
cd ~/stock-analysis
python3 scripts/taiex_rsi_divergence.py
```

輸出 CSV 至 `output/` 目錄。

## 參數調整

編輯腳本頂部的參數區塊：
- `RSI_PERIOD`: RSI 週期（預設 14）
- `LOOKBACK`: 背離偵測視窗（預設 20）
- `SYMBOL`: Yahoo Finance ticker（預設 ^TWII）
