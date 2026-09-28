# TAIEX 機器學習模型檔案備份說明 (Model Backup Specification)

## 模型檔案明細 (Model Files)
本管線採用純 NumPy 實作，模型狀態與超參數主要由 `model_hyperparams.json` 管理。在 Champion-Challenger 擂台賽中，最佳參數集會被持續更新於此處，作為所有集成模型 (Bagging, Boosting, NeuralNet) 的推論基準。

* **`model_hyperparams.json`**: 
  - 核心檔案，定義了各任務（分類/迴歸）下的模型超參數。
  - 包含 Bagging 的 `n_estimators`, `max_depth`；Boosting 的 `lr`, `n_estimators`；NeuralNet 的 `hidden`, `lr`, `epochs`。
  - 每週進化排程後，若 Challenger 勝出，此檔案將被覆蓋更新。

## 備份排程策略 (Backup Strategy)
- **頻率**: 每週一次。
- **觸發時間**: 排程在每週日/一凌晨執行完 `weekly_evolve.py` (擂台競賽與參數更新) 之後。
- **目標位置**: Google Drive `gdrive:stock-analysis/model_backups/`。
- **備份內容**: 包含 `model_hyperparams.json` 與當時的 `taiex_eval_history.csv` (評估追蹤檔)。

## 備份自動化腳本 (`backup_models.py`)
為確保流程穩固，已建立自動化備份邏輯：
```python
import subprocess
import os

def backup():
    files = ["model_hyperparams.json", "taiex_eval_history.csv"]
    target = "gdrive:stock-analysis/model_backups/"
    for f in files:
        path = os.path.expanduser(f"~/stock-analysis/data/{f}")
        if os.path.exists(path):
            subprocess.run(["rclone", "copy", path, target])
            print(f"已成功備份: {f}")
```
該腳本將在 `weekly_evolve.py` 執行結束後自動呼叫。
