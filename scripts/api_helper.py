import time
import random
import functools

def retry_on_429(max_attempts=4, initial_delay=15):
    """
    專門用來裝飾 API 請求函數的裝飾器：
    自動偵測 429 錯誤、動態拉開冷卻時間並進行指數退避重試。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    # 檢查是否為 429 流量限制
                    if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str or "yahoofinance" in err_str:
                        if attempt == max_attempts:
                            print(f"[429 錯誤] 已達最大重試次數 ({max_attempts})，放棄請求。")
                            raise e
                        
                        # 加入隨機抖動 (Jitter)
                        jitter = random.uniform(1, 5)
                        sleep_time = delay + jitter
                        print(f"[429 警告] 觸發 API 流量限制！正在執行第 {attempt}/{max_attempts} 次自動退避。")
                        print(f"-> 系統將自動進入冷卻狀態，暫停 {sleep_time:.1f} 秒後重新發起請求...")
                        
                        time.sleep(sleep_time)
                        delay *= 2  # 指數級拉開等待時間 (15s -> 30s -> 60s -> 120s)
                    else:
                        raise e
        return wrapper
    return decorator

if __name__ == "__main__":
    print("防 429 裝飾器模組載入成功。")
