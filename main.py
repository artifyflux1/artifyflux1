import os
import shutil
import time
from gradio_client import Client, handle_file

# Load list of proxies from file
with open("proxies.txt", "r") as f:
    proxies = [line.strip() for line in f if line.strip()]

result = None
client = None

# Try each proxy in sequence
for proxy in proxies:
    try:
        # Set proxy environment variables
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy

        print(f"Trying proxy: {proxy}")

        # Initialize client inside the loop to ensure new connection with proxy
        client = Client("multimodalart/wan2-1-fast")

        result = client.predict(
            input_image=handle_file('assets/influencer.png'),
            prompt="girl smiling and gaming, smooth head tilt and blinking, hands moving on controller, slow cinematic camera pan, cozy pink room lighting",
            height=1152,
            width=640,
            negative_prompt="blurry, distorted face, low quality, flickering, overexposed, motion blur, artifacts, extra limbs, deformed hands",
            duration_seconds=3.5,
            guidance_scale=1,
            steps=4,
            seed=42,
            randomize_seed=True,
            api_name="/generate_video"
        )

        print("Request succeeded with proxy:", proxy)
        break  # Exit loop if successful

    except Exception as e:
        print(f"Proxy {proxy} failed with error: {e}")
        time.sleep(2)  # Optional delay before retrying

    finally:
        # Clean up proxy environment variables
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

# Handle final outcome
if result is None:
    print("All proxies failed. Could not generate video.")
else:
    video_path = result[0]['video']
    destination_path = os.path.join(os.getcwd(), "result.mp4")
    shutil.copy(video_path, destination_path)
    print(f"Video saved to: {destination_path}")