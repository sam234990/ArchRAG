from threading import Lock
import time
import random

api_bases = [
    # "http://localhost:8876/",
    # "http://localhost:8877/",
    # "http://localhost:8878/",
    # "http://localhost:8879/",
    # "http://localhost:8880/",
    # "http://localhost:8881/",
    # "http://localhost:8882/",
    # "http://10.26.1.186:6666/",
    # "http://10.26.1.186:6667/",
    # "http://10.26.1.21:8876/",
    # "http://10.26.1.21:8877/",
    "http://localhost:11434",
]


usage_count = {base: 0 for base in api_bases}
lock = Lock()
max_usage = 4
max_retries = 5  # Maximum retry attempts
wait_time = 3  # Wait time for retries (seconds)


def get_ollama_serve_url():
    attempt = 0  # Initialize attempt counter

    while attempt < max_retries:
        with lock:
            eligible_bases = [
                base for base, count in usage_count.items() if count < max_usage
            ]
            if not eligible_bases:
                sleep_flag = True
            else:
                sleep_flag = False
                # 随机选择一个可用的 API 基地址
                api_base = random.choice(eligible_bases)
                usage_count[api_base] += 1
                return api_base
            
        if sleep_flag:
            # If all APIs are busy, wait before retrying
            time.sleep(wait_time)
            attempt += 1  # Increment attempt counter
            continue  # Retry the loop
    
    return None

def reset_ollama_serve_url(api_base):
    if api_base not in api_bases:
        return
    with lock:
        usage_count[api_base] -= 1
        return