#!/usr/bin/env python3
import os
import time
import glob
import requests
import shutil
import random

from stem import Signal
from stem.control import Controller
from gradio_client import Client, handle_file

# where Gradio dumps its temp videos (on your local root filesystem)
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
        time.sleep(TOR_REBUILD_WAIT)
        print("✅ Tor should have a new circuit now.")
    except Exception as e:
        print(f"❌ Failed to signal Tor NEWNYM: {e}")

def tor_ip_ok():
    """
    Returns True if we can fetch our IP over Tor.
    """
    try:
        resp = requests.get("https://api.ipify.org", timeout=10, 
                            proxies={
                              "http":  "socks5h://127.0.0.1:9050",
                              "https": "socks5h://127.0.0.1:9050"
                            })
        ip = resp.text
        print(f"🌐 Current exit-node IP: {ip}")
        return True
    except Exception as e:
        print(f"❌ Tor connection test failed: {e}")
        return False

def CopyVideo(output_video_path):
    TMP_DIR_PATH_MAIN = "/tmp/gradio"

    try:
        video_files = glob.glob(os.path.join(TMP_DIR_PATH_MAIN, "**", "*.mp4"), recursive=True)
        if not video_files:
            return

        video_files.sort(key=os.path.getmtime, reverse=True)
        latest_video = video_files[0]

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        shutil.copy2(latest_video, output_video_path)

        temp_folder = os.path.dirname(latest_video)
        os.remove(latest_video)
        shutil.rmtree(temp_folder)

        print(f"✅ Video copied to: {output_video_path}")
    except:
        pass  # Completely silent on any failure

def GenerateVideo(user_prompt: str,
                  negative_user_prompt: str,
                  user_input_image: str,
                  output_video: str):
    SPACE_URL = "https://multimodalart-wan2-1-fast.hf.space"

    while True:
        # 1) ensure Tor proxy env vars are set
        os.environ["HTTP_PROXY"]  = "socks5h://127.0.0.1:9050"
        os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:9050"
        os.environ["ALL_PROXY"]   = "socks5h://127.0.0.1:9050"

        if not tor_ip_ok():
            change_ip()
            continue

        # record existing gradio temp folders
        before = set(os.listdir(TMP_DIR)) if os.path.isdir(TMP_DIR) else set()

        try:
            print("🚀 Initializing Gradio client…")
            client = Client(SPACE_URL)

            print("🎬 Submitting job…")
            job = client.submit(
                input_image=handle_file(user_input_image),
                prompt=user_prompt,
                negative_prompt=negative_user_prompt,
                height=1152,
                width=640,
                duration_seconds=3.4,
                guidance_scale=1,
                steps=4,
                seed=42,
                randomize_seed=True,
                api_name="/generate_video"
            )

            print("⏳ Waiting for video to finish…")
            result = job.result(timeout=300)
            first = result[0]

            CopyVideo(output_video)
            break  # success!

        except Exception as e:
            print(f"❌ Generation/saving error: {e}")
            print("🔄 Will try again with a fresh Tor circuit…")
            change_ip()

# Declare the function to generate videos in a loop

def VideoGen(prompts_file: str, input_images_folder: str, output_videos_folder: str) -> bool:
    """
    Reads an XML-like prompts file, extracts each <image_to_video_prompt> under
    <sentence_{n}> elements, and generates a video for each sentence by calling
    GenerateVideo(user_prompt, output_image). Videos are saved in output_videos_folder
    as video_<sentence_number>.png.

    Returns True if all videos were generated successfully, False otherwise.
    """
    try:
        # Parse the prompts XML file
        tree = ET.parse(prompts_file)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError, OSError) as e:
        # Failed to read/parse the file
        print(f"Error reading prompts file: {e}")
        return False

    # Ensure the output directory exists
    try:
        os.makedirs(output_videos_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return False

    # Iterate over each sentence element
    for sentence_elem in root:
        tag = sentence_elem.tag  # e.g. "sentence_1", "sentence_2", etc.
        if not tag.startswith("sentence_"):
            # Skip any unexpected tags
            continue

        # Extract the sentence number
        try:
            sentence_number = int(tag.split("_", 1)[1])
        except (IndexError, ValueError):
            print(f"Skipping element with invalid tag: {tag}")
            continue

        # Find the video generator prompt
        prompt_elem = sentence_elem.find("image_to_video_prompt")
        if prompt_elem is None or not prompt_elem.text:
            print(f"No <image_to_video_prompt> found for {tag}, skipping.")
            continue

        user_prompt_main = prompt_elem.text.strip()

        # Construct the negative user prompt
        negative_user_prompt_from_file = sentence_elem.find("image_to_video_negative_prompt")
        if negative_user_prompt_from_file is None or not negative_user_prompt_from_file.text:
            print(f"No <image_to_video_negative_prompt> found for {tag}, skipping.")
            continue

        user_negative_prompt = negative_user_prompt_from_file.text.strip()

        # Construct the input images folder
        input_image_file_now = f"image_{sentence_number}.png"
        input_image_file_path = os.path.join(input_images_folder, input_image_file_now)

        # Construct the output video path, e.g. "/path/to/output/video_1.mp4"
        output_filename = f"video_{sentence_number}.mp4"
        output_path = os.path.join(output_videos_folder, output_filename)

        try:
            # Call the existing void function to generate and save the video
            GenerateVideo(user_prompt_main, user_negative_prompt, input_image_file_path, output_path)
        except Exception as e:
            # If any error occurs during video generation, abort
            print(f"Error generating video for {tag}: {e}")
            return False

    # If we reach here, all prompts have been processed successfully
    return True