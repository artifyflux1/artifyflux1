import requests
from urllib.parse import urlparse
import concurrent.futures
import threading

PROXY_FILE = "proxies.txt"
OUTPUT_FILE = "working_proxies.txt"
TEST_URL = "https://httpbin.org/ip"
TIMEOUT = 7

write_lock = threading.Lock()

def check_proxy(index, proxy_url):
    try:
        parsed = urlparse(proxy_url.strip())
        scheme = parsed.scheme.lower()
        if scheme not in ['http', 'https', 'socks4', 'socks5']:
            print(f"[x] [{index}] INVALID SCHEME: {proxy_url}")
            return

        proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }

        response = requests.get(TEST_URL, proxies=proxies, timeout=TIMEOUT, verify=True)
        if response.status_code == 200:
            ip = response.json().get("origin", "unknown")
            print(f"[✓] [{index}] WORKING: {proxy_url} -> {ip}")

            with write_lock:
                with open(OUTPUT_FILE, "a") as file:
                    file.write(proxy_url + "\n")
        else:
            print(f"[x] [{index}] FAILED (status): {proxy_url}")
    except Exception as e:
        print(f"[x] [{index}] FAILED ({type(e).__name__}): {proxy_url}")

def main():
    with open(PROXY_FILE, "r") as file:
        proxy_list = [line.strip() for line in file if line.strip()]

    print(f"Testing {len(proxy_list)} proxies...\n")

    open(OUTPUT_FILE, "w").close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(lambda args: check_proxy(*args), enumerate(proxy_list, start=1))

    print(f"\n✔ Done. Working proxies are written live to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
