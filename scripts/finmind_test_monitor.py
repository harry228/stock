import os
import time
import logging
import sys

sys.path.append(os.path.expanduser("~/stock-analysis/scripts"))
from finmind_connector import fetch_finmind_data

os.makedirs(os.path.expanduser("~/stock-analysis/logs"), exist_ok=True)
log_file = os.path.expanduser("~/stock-analysis/logs/finmind_test.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def test_with_backoff():
    symbols = ["2330", "2317", "0050"]
    max_attempts = 4
    delays = [5, 15, 45, 120]
    
    for symbol in symbols:
        success = False
        for attempt in range(max_attempts):
            try:
                logging.info(f"Testing FinMind API for symbol {symbol} (attempt {attempt+1}/{max_attempts})...")
                data = fetch_finmind_data(symbol, start_date="2026-08-01")
                if data and len(data) > 0:
                    logging.info(f"SUCCESS: Symbol {symbol} returned {len(data)} records. Latest: {data[-1]}")
                    success = True
                    break
                else:
                    logging.warning(f"WARNING: Symbol {symbol} returned empty data.")
            except Exception as e:
                logging.error(f"ERROR: Symbol {symbol} failed on attempt {attempt+1}: {e}")
            
            if attempt < max_attempts - 1:
                sleep_time = delays[attempt]
                logging.info(f"Backoff cooling down for {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        if not success:
            logging.critical(f"CRITICAL: Symbol {symbol} failed all {max_attempts} attempts.")
            print(f"FinMind test failed for {symbol}")
            sys.exit(1)
            
    print("FinMind API test monitor passed successfully for all symbols.")
    logging.info("FinMind API test monitor passed successfully for all symbols.")

if __name__ == "__main__":
    test_with_backoff()
