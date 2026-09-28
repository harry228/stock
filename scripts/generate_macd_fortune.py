import csv
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_path = "/data/data/com.termux/files/home/taiex_ml_training_data.csv"
output_dir = "/data/data/com.termux/files/home/stock-analysis/data"
os.makedirs(output_dir, exist_ok=True)
csv_output_path = os.path.join(output_dir, "taiex_macd_momentum_fortune_0_100.csv")
chart_output_path = os.path.join(output_dir, "taiex_macd_fortune_chart.png")

downloads_dir = "/data/data/com.termux/files/home/storage/downloads"
downloads_csv = os.path.join(downloads_dir, "taiex_macd_momentum_fortune_0_100.csv") if os.path.exists(downloads_dir) else None
downloads_chart = os.path.join(downloads_dir, "taiex_macd_fortune_chart.png") if os.path.exists(downloads_dir) else None

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

# 1. Calculate MACD (12, 26, 9)
def calc_ema(arr, span):
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    ema[0] = arr[0] # or simple average for first
    return ema

ema12 = calc_ema(closes_arr, 12)
ema26 = calc_ema(closes_arr, 26)
macd_line = ema12 - ema26
signal_line = calc_ema(macd_line, 9)
macd_hist = macd_line - signal_line

# 2. Percentage MACD Histogram to eliminate absolute price scale effect
pct_hist = macd_hist / closes_arr

# 3. Rolling normalization to 0-100 (Window = 250 days approx 1 year, fallback to expanding for initial period)
window = 250
fortune_scores = np.zeros_like(pct_hist)

for i in range(len(pct_hist)):
    start_idx = max(0, i - window + 1)
    window_data = pct_hist[start_idx:i+1]
    min_w = np.min(window_data)
    max_w = np.max(window_data)
    if max_w > min_w:
        score = (pct_hist[i] - min_w) / (max_w - min_w) * 100.0
    else:
        score = 50.0
    fortune_scores[i] = np.clip(score, 0.0, 100.0)

# Write CSV output
for path in [csv_output_path, downloads_csv]:
    if path is None: continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Close", "MACD_Hist", "Pct_Hist", "Fortune_Score_0_100"])
        for d, c, mh, ph, fs in zip(dates, closes_arr, macd_hist, pct_hist, fortune_scores):
            writer.writerow([d, f"{c:.2f}", f"{mh:.4f}", f"{ph:.6f}", f"{fs:.2f}"])
    print(f"Saved CSV: {path}")

# 4. Plotting: Top = Close Price, Bottom = Fortune Score (0-100)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top: Close Price
ax1.plot(dates, closes_arr, color='royalblue', linewidth=1.2, label='TAIEX Close')
ax1.set_title('TAIEX Close Price & MACD Momentum Fortune Score (0-100)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Index Price', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# Format x-ticks to show yearly or sparse dates
n_dates = len(dates)
step = max(1, n_dates // 8)
tick_indices = list(range(0, n_dates, step))
tick_labels = [dates[i] for i in tick_indices]

# Bottom: Fortune Score
ax2.plot(dates, fortune_scores, color='darkorange', linewidth=1.0, label='Fortune Score (0-100)')
ax2.axhline(50, color='gray', linestyle='--', alpha=0.7, label='Neutral (50)')
ax2.axhline(80, color='red', linestyle=':', alpha=0.7, label='Strong Bullish (80)')
ax2.axhline(20, color='green', linestyle=':', alpha=0.7, label='Strong Bearish (20)')
ax2.fill_between(dates, 50, fortune_scores, where=(fortune_scores >= 50), color='red', alpha=0.15)
ax2.fill_between(dates, 50, fortune_scores, where=(fortune_scores < 50), color='green', alpha=0.15)

ax2.set_ylabel('Fortune Score', fontsize=11)
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

plt.xticks(tick_indices, tick_labels, rotation=30)
plt.tight_layout()

for path in [chart_output_path, downloads_chart]:
    if path is None: continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    print(f"Saved Chart: {path}")

plt.close()
print("Generation completed successfully!")
