#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime

def backup():
    """Backup core model parameters, training data, and evaluation history to GDrive."""
    today = datetime.now().strftime("%Y%m%d")
    # 定義要備份的檔案列表與對應的本地來源路徑
    backup_files = {
        "model_hyperparams.json": os.path.expanduser("~/stock-analysis/data/model_hyperparams.json"),
        "taiex_eval_history.csv": os.path.expanduser("~/stock-analysis/data/taiex_eval_history.csv"),
        "taiex_ml_training_data.csv": os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    }
    
    target = "gdrive:stock-analysis/model_backups/"
    
    print(f"[backup_models] 開始執行模型檔案備份 (日期: {today})...")
    for original_name, path in backup_files.items():
        if os.path.exists(path):
            # 建立帶有時間戳記的新檔名
            name_part, ext = os.path.splitext(original_name)
            timestamped_name = f"{name_part}_{today}{ext}"
            
            # 將檔案拷貝到臨時位置進行重命名上傳
            tmp_path = os.path.expanduser(f"~/stock-analysis/data/{timestamped_name}")
            subprocess.run(["cp", path, tmp_path], check=True)
            
            try:
                subprocess.run(["rclone", "copy", tmp_path, target], check=True)
                print(f"[backup_models] 已成功備份: {timestamped_name}")
            except Exception as e:
                print(f"[backup_models] 備份失敗 ({original_name}): {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            print(f"[backup_models] 警告: 找不到檔案 {path}")

if __name__ == "__main__":
    backup()
