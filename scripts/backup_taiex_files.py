#!/usr/bin/env python3
import os
import shutil
import re
import csv
from datetime import datetime
import subprocess

def get_data_date():
    """
    從 taiex_future_outlook.txt 或 taiex_ml_training_data.csv 中解析最新數據基準日。
    """
    outlook_paths = [
        os.path.expanduser("~/storage/downloads/taiex_future_outlook.txt"),
        os.path.expanduser("~/stock-analysis/taiex_future_outlook.txt"),
        os.path.expanduser("~/stock-analysis/output/taiex_future_outlook.txt")
    ]
    for p in outlook_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                    m = re.search(r"【最新數據基準日】:\s*(\d{4}-\d{2}-\d{2})", content)
                    if m:
                        return m.group(1).replace("-", "")
            except Exception:
                pass

    data_paths = [
        os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv"),
        os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
    ]
    for p in data_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    reader = list(csv.DictReader(f))
                    if reader and "Date" in reader[-1]:
                        return reader[-1]["Date"].replace("-", "")
            except Exception:
                pass

    return datetime.now().strftime("%Y%m%d")

def main():
    source_dir = os.path.expanduser("~/storage/downloads")
    backup_dir = os.path.expanduser("~/Backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    date_str = get_data_date()
    print(f"解析到最新數據基準日後綴: {date_str}")
    
    files_to_backup = [
        "taiex_forecast_chart.png",
        "taiex_future_outlook.txt"
    ]
    
    for filename in files_to_backup:
        src_path = os.path.join(source_dir, filename)
        if not os.path.exists(src_path):
            alt_path = os.path.expanduser(f"~/stock-analysis/{filename}")
            if os.path.exists(alt_path):
                src_path = alt_path
            else:
                alt_path2 = os.path.expanduser(f"~/stock-analysis/output/{filename}")
                if os.path.exists(alt_path2):
                    src_path = alt_path2

        if os.path.exists(src_path):
            ext = os.path.splitext(filename)[1]
            base = os.path.splitext(filename)[0]
            new_filename = f"{base}_{date_str}{ext}"
            dst_path = os.path.join(backup_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Successfully backed up {src_path} to {dst_path}")
        else:
            print(f"Warning: Source file {filename} not found in downloads or stock-analysis folders.")

    # Sync specific files to Cloud to avoid timeout
    try:
        for filename in files_to_backup:
            ext = os.path.splitext(filename)[1]
            base = os.path.splitext(filename)[0]
            new_filename = f"{base}_{date_str}{ext}"
            src_file = os.path.join(backup_dir, new_filename)
            if os.path.exists(src_file):
                subprocess.run(["rclone", "copyto", src_file, f"gdrive:Backup/{new_filename}"], check=True)
        print("Successfully synced to GDrive.")
    except Exception as e:
        print(f"Failed to sync to GDrive: {e}")

if __name__ == "__main__":
    main()
