import os
import time
from gradio_client import Client
import requests
import re
import xml.etree.ElementTree as ET

def GenerateImage(proxy_file, image_prompt, output_image_name):
    try:
        # Load list of proxies from file
        with open(proxy_file, "r") as f:
            proxies = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: 'proxies.txt' not found.")
        return False

    if not proxies:
        print("Error: No valid proxies found in 'proxies.txt'.")
        return False

    # Try each proxy in order
    for proxy in proxies:
        try:
            # Set proxy environment variables
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy

            print(f"Trying proxy: {proxy}")

            # Initialize the Gradio client inside the loop to ensure fresh connection
            client = Client("black-forest-labs/FLUX.1-dev")

            # Generate the image via the model
            image_url = client.predict(
                prompt=image_prompt,
                seed=0,
                randomize_seed=True,
                width=1152,
                height=2048,
                guidance_scale=3.5,
                num_inference_steps=28,
                api_name="/infer"
            )

            print("Image generated successfully. Downloading...")

            # Download the image using requests (which respects proxy env vars)
            response = requests.get(image_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image. HTTP status: {response.status_code}")

            # Ensure the output directory exists
            dir_name = os.path.dirname(output_image_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Save the image to disk
            with open(output_image_name, "wb") as f:
                f.write(response.content)

            print("Image saved.")
            return True

        except Exception as e:
            print(f"Proxy '{proxy}' failed with error: {e}")
            time.sleep(2)  # Optional delay before trying next proxy

        finally:
            # Clean up proxy environment variables
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)

    # If all proxies failed
    print("All proxies failed. Could not complete the request.")
    return False

def ImageGen(prompts_file, proxy_file, output_image_folder):
    """
    Generates images for each sentence in the XML file by extracting the image generator prompt
    and calling the GenerateImage function. Returns True if all images are successfully generated,
    False otherwise.

    :param prompts_file: Path to the XML file containing image prompts.
    :param proxy_file: Proxy file to be used by GenerateImage.
    :param output_image_folder: Folder where the generated images will be saved.
    :return: Boolean indicating success or failure.
    """

    # Attempt to parse the XML file
    try:
        tree = ET.parse(prompts_file)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError, IOError):
        return False

    # Ensure the output directory exists
    try:
        os.makedirs(output_image_folder, exist_ok=True)
    except OSError:
        return False

    # Iterate through each sentence in the XML
    for sentence_element in root:
        if not sentence_element.tag.startswith('sentence_'):
            continue  # Skip non-sentence elements

        # Extract sentence number from the tag name
        parts = sentence_element.tag.split('_')
        if len(parts) != 2:
            return False  # Invalid tag format
        try:
            sentence_num = int(parts[1])
        except ValueError:
            return False  # Non-numeric suffix

        # Find the image generator prompt
        prompt_elem = sentence_element.find('image_generator_prompt')
        if prompt_elem is None or not prompt_elem.text or prompt_elem.text.strip() == '':
            return False  # Missing or empty prompt

        image_prompt = prompt_elem.text.strip()

        # Construct the output image path
        output_image_name = f"image_{sentence_num}.png"
        full_output_path = os.path.join(output_image_folder, output_image_name)

        # Call the image generation function
        success = GenerateImage(proxy_file, image_prompt, full_output_path)

        if not success:
            return False  # Stop on first failure

    return True  # All images were generated successfully

success = ImageGen(
    prompts_file="../prompts.txt",
    proxy_file="../proxies.txt",
    output_image_folder="assets/output/images"
)

if success:
    print("All images were successfully generated.")
else:
    print("An error occurred during image generation.")