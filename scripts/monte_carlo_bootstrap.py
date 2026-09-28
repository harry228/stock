#!/usr/bin/env python3
"""
RSI 背離交易系統的蒙地卡羅模擬（Monte Carlo Simulation）拔靴法（Bootstrapping）驗證
- 支援多檔個股分析
- 實現完整交易規則（停利、停損、持有天數限制）
- 進行交易層級拔靴法（Trade-level Bootstrapping）模擬資金曲線與風險指標（MDD、Sharpe）
- 進行虛無假設隨機進場顯著性測試（Null Hypothesis Significance Test），計算 p-value
- 根據 95% 蒙地卡羅最大回撤，提供動態部位調整規則（Position Sizing）
"""

import os
import sys
import math
import json
import random
import argparse
from datetime import datetime

# 確保可以匯入同目錄的 taiex_rsi_divergence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from taiex_rsi_divergence import (
        fetch_yahoo_data, load_cache, save_cache, compute_rsi, detect_divergence
    )
except ImportError:
    # 備用：若無法匯入，則直接在腳本中實現基本功能或提示
    print("警告: 無法從 taiex_rsi_divergence 匯入函數，將使用內建簡易函數。")

# 常數設定
DATA_DIR = os.path.expanduser("~/stock-analysis/data")


def percentile(data, q):
    """計算百分位數 (q 介於 0 到 100 之間)"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (q / 100.0) * (len(sorted_data) - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    weight = idx - lower
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight


def calculate_std(data, mean_val=None):
    """計算標準差"""
    n = len(data)
    if n < 2:
        return 0.0
    if mean_val is None:
        mean_val = sum(data) / n
    variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
    return math.sqrt(variance)


def backtest_single_trade(records, entry_idx, direction, tp_pct, sl_pct, hold_days):
    """
    針對單一訊號進行回測，計算該筆交易的最終報酬率。
    - direction: 1 (Long), -1 (Short)
    - 進場點：entry_idx + 1 (訊號隔日開盤)
    - 停利(TP)、停損(SL)、最大持有天數(hold_days)
    """
    n = len(records)
    if entry_idx + 1 >= n:
        return None  # 無法進場
    
    entry_price = records[entry_idx + 1]["open"]
    if entry_price <= 0:
        return None

    exit_price = None
    exit_type = "hold"  # "tp", "sl", "hold"
    exit_idx = entry_idx + 1

    # 決定停損停利價格
    if direction == 1:  # Long
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
    else:  # Short
        tp_price = entry_price * (1.0 - tp_pct / 100.0)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)

    # 模擬持倉期間
    end_idx = min(entry_idx + hold_days, n - 1)
    for i in range(entry_idx + 1, end_idx + 1):
        high = records[i]["high"]
        low = records[i]["low"]
        close = records[i]["close"]
        exit_idx = i

        if direction == 1:  # Long
            # 檢查是否觸及停損或停利
            if low <= sl_price and high >= tp_price:
                # 極端情況：同日觸及停利與停損，保守以停損計
                exit_price = sl_price
                exit_type = "sl"
                break
            elif low <= sl_price:
                exit_price = sl_price
                exit_type = "sl"
                break
            elif high >= tp_price:
                exit_price = tp_price
                exit_type = "tp"
                break
        else:  # Short
            if high >= sl_price and low <= tp_price:
                exit_price = sl_price
                exit_type = "sl"
                break
            elif high >= sl_price:
                exit_price = sl_price
                exit_type = "sl"
                break
            elif low <= tp_price:
                exit_price = tp_price
                exit_type = "tp"
                break

    # 若未觸及停利停損，最後一天以收盤價出場
    if exit_price is None:
        exit_price = records[end_idx]["close"]
        exit_type = "hold"

    # 計算報酬率 %
    if direction == 1:
        ret = (exit_price - entry_price) / entry_price * 100.0
    else:
        ret = (entry_price - exit_price) / entry_price * 100.0

    return {
        "entry_date": records[entry_idx + 1]["date"],
        "exit_date": records[exit_idx]["date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "type": exit_type,
        "return": round(ret, 4),
        "hold_days": exit_idx - entry_idx
    }


def run_strategy_backtest(records, signals, tp_pct, sl_pct, hold_days):
    """對所有訊號執行完整回測，回傳交易清單"""
    trades = []
    for i, sig in enumerate(signals):
        if sig == 0:
            continue
        trade = backtest_single_trade(records, i, sig, tp_pct, sl_pct, hold_days)
        if trade:
            trade["signal_idx"] = i
            trade["direction"] = "Long" if sig == 1 else "Short"
            trades.append(trade)
    return trades


def bootstrap_equity_curves(trade_returns, num_simulations=5000, initial_capital=100.0):
    """
    對實際交易報酬率進行拔靴法抽樣，模擬資金曲線。
    回傳各模擬路徑的統計結果：[ { 'final_return': f, 'mdd': m, 'win_rate': w, 'sharpe': s }, ... ]
    """
    results = []
    n_trades = len(trade_returns)
    if n_trades == 0:
        return results

    for _ in range(num_simulations):
        # 有放回抽樣
        sim_returns = [random.choice(trade_returns) for _ in range(n_trades)]
        
        # 計算資金曲線與最大回撤
        equity = initial_capital
        equity_curve = [equity]
        peaks = [equity]
        win_count = 0
        
        for r in sim_returns:
            # 複利計算
            equity = equity * (1.0 + r / 100.0)
            equity_curve.append(equity)
            peaks.append(max(peaks[-1], equity))
            if r > 0:
                win_count += 1
                
        # 計算最大回撤 (MDD)
        mdd = 0.0
        for i in range(len(equity_curve)):
            dd = (peaks[i] - equity_curve[i]) / peaks[i] * 100.0
            if dd > mdd:
                mdd = dd
                
        final_return = (equity - initial_capital) / initial_capital * 100.0
        win_rate = win_count / n_trades * 100.0
        
        # 計算 Sharpe Ratio (以每筆交易回報為基準)
        avg_r = sum(sim_returns) / n_trades
        std_r = calculate_std(sim_returns, avg_r)
        # 假設無風險利率為 0，Sharpe = 平均報酬 / 標準差 (若標準差為0則為0)
        sharpe = (avg_r / std_r) if std_r > 0 else 0.0

        results.append({
            "final_return": final_return,
            "mdd": mdd,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "sim_returns": sim_returns
        })
        
    return results


def run_null_significance_test(records, rsi_period, lookback, num_long, num_short, tp_pct, sl_pct, hold_days, null_sims=1000):
    """
    虛無假設顯著性測試 (Random Entry Bootstrap)
    - 隨機選擇交易日進場，並隨機分配 Long/Short
    - 計算 1000 次隨機交易組合的平均報酬
    - 回傳隨機平均報酬數組，用於計算實際策略的 p-value
    """
    n = len(records)
    # 確保進場點在指標計算完畢後，且有足夠時間出場
    valid_indices = list(range(rsi_period + lookback, n - hold_days - 2))
    total_signals = num_long + num_short
    
    if len(valid_indices) < total_signals or total_signals == 0:
        return []

    null_avg_returns = []
    
    for _ in range(null_sims):
        # 隨機挑選進場位置
        rand_indices = random.sample(valid_indices, total_signals)
        # 前 num_long 個為 Long，後 num_short 個為 Short
        sim_returns = []
        for idx, entry_idx in enumerate(rand_indices):
            direction = 1 if idx < num_long else -1
            trade = backtest_single_trade(records, entry_idx, direction, tp_pct, sl_pct, hold_days)
            if trade:
                sim_returns.append(trade["return"])
        
        if sim_returns:
            null_avg_returns.append(sum(sim_returns) / len(sim_returns))
            
    return null_avg_returns


def load_data_for_symbol(symbol, years):
    """載入或下載指定個股資料"""
    cache_path = os.path.join(DATA_DIR, f"{symbol.replace('^','')}_cache.json")
    
    # 嘗試載入快取
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            records = cache.get("records", [])
            if records:
                # 簡單檢查快取是否太舊（大於1天），是則增量更新
                last_date = records[-1]["date"]
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                if (datetime.now() - last_dt).days > 1:
                    print(f"[{symbol}] 快取最後日期為 {last_date}，嘗試增量更新...")
                    new_records = fetch_yahoo_data(symbol, start_date=last_date)
                    if new_records:
                        date_map = {r["date"]: r for r in records}
                        for r in new_records:
                            date_map[r["date"]] = r
                        records = sorted(date_map.values(), key=lambda x: x["date"])
                        save_cache(records, symbol)
                else:
                    print(f"[{symbol}] 載入快取: {len(records)} 筆 ({records[0]['date']} ~ {records[-1]['date']})")
                return records
        except Exception as e:
            print(f"[{symbol}] 載入快取失敗: {e}，將重新下載。")

    # 重新下載
    try:
        records = fetch_yahoo_data(symbol, years=years)
        if records:
            save_cache(records, symbol)
            return records
    except Exception as e:
        print(f"[{symbol}] 下載失敗: {e}")
    return []


def analyze_symbol(symbol, years, tp_pct, sl_pct, hold_days, mc_sims, null_sims, risk_limit):
    """分析單一個股並列印結果"""
    print("=" * 80)
    print(f" 正在分析標的: {symbol} ")
    print("=" * 80)
    
    records = load_data_for_symbol(symbol, years)
    if not records:
        print(f"錯誤: 無法取得 {symbol} 的日線資料，跳過。")
        return None

    # 1. 計算 RSI
    closes = [r["close"] for r in records]
    rsi_period = 14
    rsi_values = compute_rsi(closes, rsi_period)
    
    # 2. 偵測背離訊號 (使用建議參數：extreme_window=5, min_interval=5, max_dist=60)
    lookback = 20
    signals = detect_divergence(
        records, rsi_values, lookback=lookback,
        extreme_window=5, min_interval=5, max_dist=60
    )
    
    # 3. 執行回測取得每筆交易報酬率
    trades = run_strategy_backtest(records, signals, tp_pct, sl_pct, hold_days)
    num_trades = len(trades)
    
    if num_trades == 0:
        print(f"警告: {symbol} 在此期間未偵測到任何 RSI 背離買賣訊號。")
        return {
            "symbol": symbol,
            "num_trades": 0
        }

    long_trades = [t for t in trades if t["direction"] == "Long"]
    short_trades = [t for t in trades if t["direction"] == "Short"]
    num_long = len(long_trades)
    num_short = len(short_trades)
    
    trade_returns = [t["return"] for t in trades]
    avg_return = sum(trade_returns) / num_trades
    win_rate = len([r for r in trade_returns if r > 0]) / num_trades * 100.0
    
    # 計算最大獲利/最大回撤
    max_profit = max(trade_returns) if trade_returns else 0.0
    max_loss = min(trade_returns) if trade_returns else 0.0
    
    # 計算勝/敗交易平均
    pos_returns = [r for r in trade_returns if r > 0]
    neg_returns = [r for r in trade_returns if r < 0]
    avg_win = sum(pos_returns) / len(pos_returns) if pos_returns else 0.0
    avg_loss = sum(neg_returns) / len(neg_returns) if neg_returns else 0.0
    profit_factor = (sum(pos_returns) / abs(sum(neg_returns))) if neg_returns else float('inf')

    print(f"【歷史回測統計】")
    print(f"  總交易次數: {num_trades} 次 (多單: {num_long} 次 | 空單: {num_short} 次)")
    print(f"  回測勝率:   {win_rate:.2f}%")
    print(f"  平均報酬率: {avg_return:+.2f}%")
    print(f"  獲利因子:   {profit_factor:.2f}")
    print(f"  平均獲利:   {avg_win:+.2f}% | 平均虧損: {avg_loss:+.2f}%")
    print(f"  最大單筆獲利: {max_profit:+.2f}% | 最大單筆虧損: {max_loss:+.2f}%")
    print(f"  交易離場類型分佈: ")
    tp_cnt = len([t for t in trades if t["type"] == "tp"])
    sl_cnt = len([t for t in trades if t["type"] == "sl"])
    hold_cnt = len([t for t in trades if t["type"] == "hold"])
    print(f"    - 停利離場 (TP): {tp_cnt} 次 ({tp_cnt/num_trades*100:.1f}%)")
    print(f"    - 停損離場 (SL): {sl_cnt} 次 ({sl_cnt/num_trades*100:.1f}%)")
    print(f"    - 到期離場 (Hold): {hold_cnt} 次 ({hold_cnt/num_trades*100:.1f}%)")

    # 4. 蒙地卡羅資金曲線與風險拔靴法 (Trade-Level Bootstrap)
    print(f"\n【蒙地卡羅拔靴法模擬結果 (重複 {mc_sims} 次)】")
    mc_results = bootstrap_equity_curves(trade_returns, num_simulations=mc_sims)
    
    final_returns = [r["final_return"] for r in mc_results]
    mdds = [r["mdd"] for r in mc_results]
    win_rates = [r["win_rate"] for r in mc_results]
    sharpes = [r["sharpe"] for r in mc_results]
    
    mean_final_ret = sum(final_returns) / mc_sims
    p5_final_ret = percentile(final_returns, 5)
    p50_final_ret = percentile(final_returns, 50)
    p95_final_ret = percentile(final_returns, 95)
    
    mean_mdd = sum(mdds) / mc_sims
    p95_mdd = percentile(mdds, 95)  # 95% 置信度下的最大回撤 (VaR MDD)
    p50_mdd = percentile(mdds, 50)
    p5_mdd = percentile(mdds, 5)
    
    mean_wr = sum(win_rates) / mc_sims
    mean_sharpe = sum(sharpes) / mc_sims
    
    # 破產率 (Probability of Ruin - 假設資金下跌超過 30%)
    ruin_count = sum(1 for r in mdds if r >= 30.0)
    prob_ruin = ruin_count / mc_sims * 100.0
    # 獲利機率
    prob_profit = sum(1 for r in final_returns if r > 0.0) / mc_sims * 100.0

    print(f"  指標項目              |   平均值   |  5% 分位數 | 中位數(50%)| 95% 分位數")
    print(f"  ----------------------+------------+------------+------------+-----------")
    print(f"  模擬總報酬率 (%)      |  {mean_final_ret:>+8.2f}% |  {p5_final_ret:>+8.2f}% |  {p50_final_ret:>+8.2f}% |  {p95_final_ret:>+8.2f}%")
    print(f"  模擬最大回撤 (MDD %)  |  {mean_mdd:>8.2f}% |  {p5_mdd:>8.2f}% |  {p50_mdd:>8.2f}% |  {p95_mdd:>8.2f}%")
    print(f"  ----------------------+------------+------------+------------+-----------")
    print(f"  模擬平均勝率: {mean_wr:.2f}% | 模擬平均 Sharpe: {mean_sharpe:.3f}")
    print(f"  策略最終獲利機率: {prob_profit:.1f}%")
    print(f"  策略破產機率 (回撤>=30%): {prob_ruin:.1f}%")

    # 5. 虛無假設隨機進場顯著性測試 (Null Hypothesis Significance Test)
    print(f"\n【虛無假設顯著性測試 (隨機交易模擬 {null_sims} 次)】")
    null_avg_returns = run_null_significance_test(
        records, rsi_period, lookback, num_long, num_short, tp_pct, sl_pct, hold_days, null_sims=null_sims
    )
    
    p_value = 1.0
    if null_avg_returns:
        # 計算 p-value: 隨機平均報酬高於或等於實際平均報酬的比例
        better_sims = sum(1 for r in null_avg_returns if r >= avg_return)
        p_value = better_sims / len(null_avg_returns)
        mean_null_ret = sum(null_avg_returns) / len(null_avg_returns)
        p95_null_ret = percentile(null_avg_returns, 95)
        
        print(f"  隨機交易平均報酬率: {mean_null_ret:+.2f}% (95% 隨機上限: {p95_null_ret:+.2f}%)")
        print(f"  實際策略平均報酬率: {avg_return:+.2f}%")
        
        sig_status = "顯著通過 (PASS)" if p_value < 0.05 else "不顯著 (FAIL)"
        p_color = "\033[92m" if p_value < 0.05 else "\033[91m"
        reset_color = "\033[0m"
        # 為了 terminal 顯示，我們在這裡做乾淨的文字輸出
        print(f"  統計顯著性 p-value :  {p_value:.4f}  →  {sig_status}")
        print(f"  (註：p-value < 0.05 代表策略在統計上顯著優於隨機進場，非市場隨機波動造成的幸運結果)")
    else:
        print("  無法進行顯著性測試 (可能有效交易日區間過小)")

    # 6. 蒙地卡羅資金風控部位調整規則 (Position Sizing Rules)
    print(f"\n【蒙地卡羅資金部位控制規則】")
    print(f"  使用者設定最大可接受總回撤: {risk_limit:.1f}%")
    print(f"  95% 置信度最大回撤 (MC-MDD 95%): {p95_mdd:.2f}%")
    
    if p95_mdd > 0:
        suggested_size_ratio = risk_limit / p95_mdd
        print(f"  推薦部位曝險比例 (Leverage/Sizing Factor): {suggested_size_ratio:.2f}x")
        if suggested_size_ratio < 0.5:
            print("  [風控建議] ✦ 此標的波動及回撤風險極高，建議將單筆部位調降至 30% 以下，或暫不操作。")
        elif suggested_size_ratio < 1.0:
            print(f"  [風控建議] ✦ 建議將部位縮減至標準部位的 {suggested_size_ratio*100:.1f}%，以防範極端回撤。")
        else:
            print(f"  [風控建議] ✦ 該標的風險受控，可使用標準 100% 部位或最大 {min(2.0, suggested_size_ratio):.2f} 倍槓桿操作。")
    else:
        suggested_size_ratio = 1.0
        print("  推薦部位比例: 無法計算 (無有效 MDD)")

    return {
        "symbol": symbol,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "profit_factor": profit_factor,
        "mc_mean_ret": mean_final_ret,
        "mc_p5_ret": p5_final_ret,
        "mc_p95_mdd": p95_mdd,
        "p_value": p_value,
        "suggested_size_ratio": suggested_size_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="RSI 背離交易系統的蒙地卡羅拔靴法模擬驗證")
    parser.add_argument(
        "--symbols", type=str, default="^TWII,2330.TW,0050.TW",
        help="分析標的列表，以逗號分隔（例如: 2330.TW,2454.TW,^TWII）"
    )
    parser.add_argument(
        "--years", type=int, default=10,
        help="若無快取時，從 Yahoo Finance 下載的歷史資料年數（預設: 10）"
    )
    parser.add_argument(
        "--tp", type=float, default=10.0,
        help="交易規則：停利百分比 (Take Profit %%) (預設: 10.0)"
    )
    parser.add_argument(
        "--sl", type=float, default=5.0,
        help="交易規則：停損百分比 (Stop Loss %%) (預設: 5.0)"
    )
    parser.add_argument(
        "--hold", type=int, default=20,
        help="交易規則：最大持股天數 (Hold Days) (預設: 20)"
    )
    parser.add_argument(
        "--sims", type=int, default=5000,
        help="蒙地卡羅資金曲線抽樣次數 (預設: 5000)"
    )
    parser.add_argument(
        "--null-sims", type=int, default=1000,
        help="虛無假設顯著性測試隨機模擬次數 (預設: 1000)"
    )
    parser.add_argument(
        "--risk-limit", type=float, default=10.0,
        help="使用者可忍受的最大資金回撤限額 pct（預設: 10.0）"
    )
    
    args = parser.parse_args()
    
    symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    print("=" * 80)
    print("                RSI 背離交易系統 — 蒙地卡羅拔靴法與顯著性驗證")
    print(f" 執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 交易規則參數：停利 +{args.tp}% | 停損 -{args.sl}% | 最長持有 {args.hold} 天")
    print(f" 模擬設定：資金曲線模擬 {args.sims} 次 | 隨機進場模擬 {args.null_sims} 次")
    print("=" * 80)
    
    all_results = []
    for sym in symbols_list:
        res = analyze_symbol(
            symbol=sym,
            years=args.years,
            tp_pct=args.tp,
            sl_pct=args.sl,
            hold_days=args.hold,
            mc_sims=args.sims,
            null_sims=args.null_sims,
            risk_limit=args.risk_limit
        )
        if res:
            all_results.append(res)
            
    # ===== 多個股比較摘要表 =====
    if len(all_results) > 1:
        print("\n" + "=" * 95)
        print("                               多股對比及風控決策摘要表")
        print("=" * 95)
        print(f"  {'標代號':<10} | {'訊號數':>6} | {'回測勝率':>8} | {'平均報酬':>8} | {'p-value':>8} | {'95%MC-MDD':>9} | {'推薦部位曝險':>12}")
        print("-" * 95)
        for r in all_results:
            if r.get("num_trades", 0) == 0:
                print(f"  {r['symbol']:<10} | {'0':>6} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9} | {'暫不交易':>12}")
                continue
            
            p_val_str = f"{r['p_value']:.4f}"
            # 標註顯著性
            if r['p_value'] < 0.05:
                p_val_str += "*"
                
            mdd_str = f"{r['mc_p95_mdd']:.2f}%"
            size_str = f"{r['suggested_size_ratio']:.2f}x"
            
            print(f"  {r['symbol']:<10} | {r['num_trades']:>6d} | {r['win_rate']:>7.2f}% | {r['avg_return']:>+7.2f}% | {p_val_str:>8} | {mdd_str:>9} | {size_str:>12}")
        print("-" * 95)
        print("  註：p-value 欄位有 * 者代表通過 95% 顯著性檢定 (p < 0.05)。推薦部位曝險為相對於 risk_limit 的調整乘數。")
        print("=" * 95)


if __name__ == "__main__":
    main()
