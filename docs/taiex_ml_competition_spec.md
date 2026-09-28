# 台股 ML 自我進化競賽與模型架構規格說明書 (TAIEX ML Competition & Architecture Specification)

> **版本 (Version)**: 1.0  
> **更新日期 (Date)**: 2026-08-23  
> **環境需求**: Termux / Android 輕量無重依賴環境 (Pure Python + NumPy)  
> **備份路徑**: Google Drive `gdrive:stock-analysis/docs/taiex_ml_competition_spec.md`

---

## 一、 模型架構與機器學習演算法 (ML Model Architectures)

為適應行動端 (Termux/Android) 設備資源限制並避免複雜 C/C++ 編譯套件依賴，本管線採用純 **NumPy** 自研的高效機器學習演算法，涵蓋四大模型種類，進行 **雙任務 (Dual-Task)** 預測（三相趨勢分類 + 未來收盤價比率迴歸）：

1. **Bagging (Random Forest / 隨機森林)**
   - **機制**: 透過 Bootstrapping 對訓練集做樣本抽樣，並在特徵空間做隨機子集抽樣。
   - **基底**: 多棵 `PureTree` 決策樹（預設 `max_depth=5`, `n_estimators=30`）。
   - **優點**: 抗過擬合能力強，能平滑單一樹的方差 (Variance)。

2. **Boosting (GBDT / 梯度提升決策樹)**
   - **機制**: 採用梯度提升原理，逐代 (Sequential) 建立淺層決策樹 / 決策墩 (Decision Stumps)，每一代以學習率 ($\eta=0.08$) 擬合上一代的殘差 (Residuals)。
   - **分類**: 使用 Cross-Entropy Loss 梯度更新；**迴歸**: 使用 MSE Loss 梯度更新。
   - **優點**: 對複雜邊界與非線性趨勢具有極高的偏差 (Bias) 修正能力。

3. **NeuralNet (Temporal MLP / 時序多層感知機)**
   - **機制**: 純 NumPy 實作的前饋神經網路（隱藏層 32 神經元，包含 Admin/ReLU 激活與 Softmax / Linear 輸出）。
   - **優點**: 能跨特徵維度捕捉高階非線性交互作用 (High-order Interaction)。

4. **Stacking (Meta-Learner / 後設集成學習器)**
   - **機制**: 採用二階 Ensemble 架構，將 Bagging、Boosting、NeuralNet 的預測機率/比率組合成 Meta-Features，透過 Logistic/Linear Meta-Learner 計算動態權重，輸出最終預測結果。
   - **用途**: 線上推論 (`infer_multi_task_ensemble.py`) 與日常預測追蹤發布的核心模型。

---

## 二、 資料數據與平穩特徵庫 (Data & Feature Engineering)

### 1. 嚴格平穩性原則 (Strict Stationarity Principle)
* **嚴禁輸入原始價格/指數點數**（如 15000 或 24000 點）：絕對價格隨時間大幅飄移，若將絕對點數餵給模型，模型會在未來未見過的價格區間失效。
* **全特徵標準化/比例化**：所有特徵必須轉換為無量綱的比率 (Ratio)、百分比 (Percentage) 或固定區間擺盪指標。

### 2. 現行 18 項核心平穩特徵明細 (Core Stationary Features)
| 分類 | 特徵名稱 (Feature Name) | 計算公式與說明 |
| :--- | :--- | :--- |
| **價格相對比率** | `Open_Ratio`, `High_Ratio`, `Low_Ratio` | 當日開/高/低價相對於當日 Close 的比率 ($\frac{Price}{Close}$) |
| **量能動能** | `Volume_Ratio_5`, `Volume_Ratio_20`, `Volume_Ratio_60` | 當日成交量相對於 5日/20日/60日 移動平均量的比率 |
| **均線乖離率 (BIAS)** | `BIAS_3`, `BIAS_5`, `BIAS_10`, `BIAS_20`, `BIAS_60`, `BIAS_120` | 收盤價相對於各天數 SMA 的偏離百分比 ($\frac{Close - SMA_n}{SMA_n} \times 100\%$) |
| **通道與波動度** | `BB_Width`, `BB_PercentB` | 布林通道寬度 ($\frac{Upper - Lower}{Middle}$) 與 %B 著陸位置 |
| **擺盪指標** | `K`, `D` | 9,3,3 隨機指標 (KD) 數值 (區間 0~100) |
| **動能與背離** | `RSI`, `Divergence_Signal` | 14日相對強弱指標與 RSI 價量背離演算法觸發訊號 (-1, 0, 1) |

---

## 三、 資料篩選與淘汰機制 (Feature Selection & Screening Rules)

為確保模型特徵簡練，防止多重共線性 (Multicollinearity) 與維度災難，採取以下三層篩選與淘汰評比：

1. **第一層：平穩性嚴格過濾 (Stationarity Test)**
   - 任何未經比例化、含有 Trend/Level 的原始數據（如絕對股價、原始成交量、指數點數）一律在資料準備階段 (`generate_ml_labels.py`) 直接淘汰，禁止進入模型訓練集。

2. **第二層：高共線性與關聯性評比 (Correlation Matrix Screening)**
   - **相關係數判定 ($r > 0.95$)**：評估特徵間的皮爾森相關係數 (Pearson Correlation)。若兩特徵相關性過高（例如 `BIAS_5` 與 `BIAS_3` 高度重疊），且其中一者對目標變數的互資訊量 (Mutual Information) 或樹分裂貢獻度顯著偏低，則進行剔除或合併。
   - **目的**: 避免多個相同特徵稀釋 Decision Tree / NeuralNet 的特徵選擇權重。

3. **第三層：方差與有效資訊過濾 (Variance & Null Warmup Filter)**
   - 移除方差接近 0 的常數特徵。
   - 自動裁切指標計算熱身期（如 120日 均線所需的前 120 個交易日數據），確保進入模型的每一筆資料皆無 NaN / Null 缺值。

---

## 四、 新資料加入理由與審核流程 (Rationale & Protocol for New Features)

當未來欲引入新數據（如外資買賣超、USD/TWD 匯率、美股指數、黃金/黃豆動能）時，必須遵守以下標準流程，避免無意義的特徵膨脹：

1. **經濟學與總經邏輯支撐 (Economic Rationality)**
   - 新資料必須具備明確的市場傳導機制。例如：
     - **USD/TWD 匯率**: 反映外資資金進出台股大盤的領先資金流指標。
     - **黃金/黃豆動能**: 反映全球通膨預期與市場避險情緒 (Risk-Off / Risk-On)。

2. **平穩化轉換要求 (Stationary Transformation)**
   - 外部數據加入前，必須轉換為：日報酬率 (Daily Return)、動能 z-score、或對數差分比率。嚴禁直接輸入匯率絕對值或商品期貨絕對價格。

3. **A/B 擂台驗證關卡 (Walk-Forward Tournament Promotion Rule)**
   - 新特徵加入後，必須在 `weekly_evolve.py` 每週擂台賽中進行 80/20 時間序列 Walk-Forward 驗證。
   - **晉級門檻**：加入新特徵的 Challenger 模型，其驗證集 **MAE 必須比現行 Champion 低 1.5% 以上**，且 **Macro F1 / 方向準確率未惡化**，方可被批准寫入 `model_hyperparams.json` 並正式併入特徵庫。

---

## 五、 自我進化評估與風險指標 (Tournament Evaluation Metrics)

競賽與進化報告 (`taiex_self_evolution_weekly_report.txt`) 針對 1d, 2d, 3d, 4d, 5d 多時間級別進行全面評估：

* **分類指標**: Accuracy (準確率), Macro F1-Score, 3x3 混淆矩陣 (Confusion Matrix)
* **迴歸指標**: MAE (平均絕對誤差), RMSE (均方根誤差), $R^2$ (決定係數)
* **實盤風險指標**:
  - **夏普比率 (Sharpe Ratio)**: 評估單位風險下的超額報酬。
  - **最大回撤 (Max Drawdown, MDD)**: 評估資產最大可能跌幅。
  - **卡瑪比率 (Calmar Ratio)**: 年化報酬率與 MDD 之比率，確保模型具備風控防禦力。

---

*此文件已同步儲存於本地與 Google Drive 雲端，作為日後新增特徵、調整模型超參數與防止重複開發之規格依據。*
