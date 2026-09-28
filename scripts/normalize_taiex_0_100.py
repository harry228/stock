import csv
import numpy as np
import os

input_path = "/data/data/com.termux/files/home/taiex_ml_training_data.csv"
output_dir = "/data/data/com.termux/files/home/stock-analysis/data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "taiex_normalized_0_100.csv")

# Also copy/save to downloads if accessible
downloads_dir = "/data/data/com.termux/files/home/storage/downloads"
downloads_path = os.path.join(downloads_dir, "taiex_normalized_0_100.csv") if os.path.exists(downloads_dir) else None

dates = []
closes = []

with open(input_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        date_str = r.get("Date", "").strip()
        close_str = r.get("Close", "").strip()
        if date_str and close_str:
            dates.append(date_str)
            closes.append(float(close_str))

closes_arr = np.array(closes)
min_val = np.min(closes_arr)
max_val = np.max(closes_arr)

# MinMax scaling to [0, 100]
if max_val > min_val:
    norm_closes = (closes_arr - min_val) / (max_val - min_val) * 100.0
else:
    norm_closes = np.zeros_like(closes_arr)

# Write output CSV
for path in [output_path, downloads_path]:
    if path is None:
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Close", "Normalized_Close_0_100"])
        for d, c, nc in zip(dates, closes_arr, norm_closes):
            writer.writerow([d, f"{c:.2f}", f"{nc:.4f}"])
    print(f"Saved: {path}")

print(f"Total rows: {len(dates)}")
print(f"Min Close: {min_val:.2f}, Max Close: {max_val:.2f}")
