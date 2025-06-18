import os
import shutil
import subprocess
import time
import requests
import random

from stem import Signal
from stem.control import Controller
from gradio_client import Client, handle_file

# where Gradio dumps its temp images
TMP_DIR = os.path.join(os.path.sep, "tmp", "gradio")

# how long to wait after NEWNYM for Tor circuit to rebuild
TOR_REBUILD_WAIT = 5

def change_ip():
    """
    Signal Tor for a new circuit (NEWNYM) and wait a bit.
    """
    pw = "this_is_my_password"
    if not pw:
        print("⚠️ TOR_CONTROL_PASSWORD not set — cannot change IP")
        return

    print("🔄 Requesting new Tor identity…")
    try:
        with Controller.from_port(port=9051) as ctrl:
            ctrl.authenticate(password=pw)
            ctrl.signal(Signal.NEWNYM)
        # give Tor clients time to rebuild circuits
        time.sleep(TOR_REBUILD_WAIT)
        print("✅ Tor should have a new circuit now.")
    except Exception as e:
        print(f"❌ Failed to signal Tor NEWNYM: {e}")

def tor_ip_ok():
    """
    Returns True if we can fetch our IP over Tor.
    """
    try:
        ip = requests.get("https://api.ipify.org", timeout=10).text
        print(f"🌐 Current exit-node IP: {ip}")
        return True
    except Exception as e:
        print(f"❌ Tor connection test failed: {e}")
        return False

import os
import shutil

def delete_gradio_tmp():
    target_dir = "/tmp/gradio"
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def GenerateImage(user_prompt: str, output_image: str):
    SPACE_URL = "https://bytedance-hyper-flux-8steps-lora.hf.space"

    while True:
        # 1) ensure Tor is running & our proxy env vars are set
        os.environ["HTTP_PROXY"]  = "socks5h://127.0.0.1:9050"
        os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:9050"
        os.environ["ALL_PROXY"]   = "socks5h://127.0.0.1:9050"

        if not tor_ip_ok():
            change_ip()
            continue

        # record existing gradio subdirs
        try:
            before = set(os.listdir(TMP_DIR))
        except FileNotFoundError:
            before = set()

        try:
            print("🚀 Initializing Gradio client…")
            client = Client(SPACE_URL)

            seed = random.randint(1000, 9999)
            print(f"🎬 Submitting job (seed={seed})…")
            job = client.submit(
                height=1152,
                width=648,
                steps=8,
                scales=3.5,
                prompt=user_prompt,
                seed=seed,
                api_name="/process_image"
            )

            print("⏳ Waiting for image to finish…")
            result = job.result(timeout=300)

            # figure out where the image really lives
            source_path = None
            first = result[0]

            # Case A: dict token → download it
            if isinstance(first, dict) and "image" in first:
                token = first["image"]
                print("📥 Downloading image token via handle_file…")
                tmp = handle_file(token)
                source_path = tmp.name

            # Case B: direct file path string that exists
            elif isinstance(first, str) and first.lower().endswith(".png") and os.path.isfile(first):
                print(f"📂 Gradio gave us a file: {first}")
                source_path = first

            # Case C: fallback → scan /tmp/gradio for brand-new subdir
            else:
                print("🔍 Falling back to scan /tmp/gradio for new image…")
                time.sleep(1)  # let it flush
                try:
                    after = set(os.listdir(TMP_DIR))
                except FileNotFoundError:
                    after = set()
                new = sorted(after - before,
                             key=lambda d: os.path.getmtime(os.path.join(TMP_DIR, d)),
                             reverse=True)
                if not new:
                    raise RuntimeError(f"No new folder under {TMP_DIR}")
                newest = new[0]
                candidate = os.path.join(TMP_DIR, newest, "image.png")
                if not os.path.isfile(candidate):
                    raise FileNotFoundError(f"Expected image at {candidate}")
                print(f"📂 Found {candidate}")
                source_path = candidate

            # sanity check
            if not source_path or not os.path.isfile(source_path):
                raise RuntimeError(f"Could not locate generated image (tried {source_path})")

            # copy into cwd, creating directories if needed
            dest = os.path.join(os.getcwd(), output_image)
            dest_dir = os.path.dirname(dest)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)

            shutil.copy(source_path, dest)
            print(f"✅ Image saved to: {dest}")
            # Clear out the directory
            delete_gradio_tmp()
            break  # done!

        except Exception as e:
            print(f"❌ Generation/saving error: {e}")
            # try a fresh IP next iteration
            change_ip()
