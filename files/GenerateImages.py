import os
import time
import requests
from gradio_client import Client

# Load list of proxies from file
with open("proxies.txt", "r") as f:
    proxies = [line.strip() for line in f if line.strip()]

# Variable to store final image URL
image_url = None

# Try each proxy in order
for proxy in proxies:
    try:
        # Set proxy environment variables
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy

        print(f"Attempting request with proxy: {proxy}")

        # Initialize the Gradio client inside the loop to ensure fresh connection
        client = Client("black-forest-labs/FLUX.1-dev")

        # Generate the image via the model
        image_url = client.predict(
            prompt="Hello!!",
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            api_name="/infer"
        )

        print("Image generated successfully. Downloading...")

        # Download the image using requests (which respects proxy env vars)
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image. HTTP status: {response.status_code}")

        # Save the image to disk
        with open("result.png", "wb") as f:
            f.write(response.content)

        print("Image saved as result.png")
        break  # Exit loop on success

    except Exception as e:
        print(f"Proxy '{proxy}' failed with error: {e}")
        time.sleep(2)  # Optional delay before trying next proxy

    finally:
        # Clean up proxy environment variables
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

else:
    # This block runs only if the loop completes without a break
    print("All proxies failed. Could not complete the request.")