import os
import time
from gradio_client import Client
import requests
import re

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

current_index = 0

def ExtractPrompt(file_path):
    """
    Extracts the content between the nth <image_generator_prompt> and </image_generator_prompt> tags
    from the specified file. Each call returns the next prompt in sequence.

    Parameters:
        file_path (str): Path to the text file containing the prompts.

    Returns:
        str: The content of the nth <image_generator_prompt> block.

    Raises:
        IndexError: If there are no more prompts to return.
        FileNotFoundError: If the specified file does not exist.
        Exception: For other unexpected errors.
    """
    global current_index

    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Compile regex pattern once per call
        prompt_pattern = re.compile(
            r'<image_generator_prompt>(.*?)</image_generator_prompt>',
            re.DOTALL
        )

        # Extract all prompts
        prompts = prompt_pattern.findall(text)

        # Check if index is within bounds
        if current_index >= len(prompts):
            raise IndexError("No more prompts available in the file.")

        # Return the current prompt and increment the index
        prompt = prompts[current_index].strip()
        current_index += 1
        return prompt

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{file_path}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

def TotalSentences(input_file_path):
    """
    Counts the number of <sentence_n> opening tags in the given input file.

    Parameters:
        input_file_path (str): Path to the input file containing XML-like data.

    Returns:
        int: The total number of <sentence_n> tags found.
    """
    count = 0

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith('<sentence_') and stripped_line.endswith('>') and not stripped_line.startswith('</sentence_'):
                count += 1

    return count

def generate_image(proxy_file_path, image_generation_prompt, output_image_name) -> bool:
    print(proxy_file_path)
    print(image_generation_prompt)
    print(output_image_name)
    return True

def generate_images(proxy_file_path: str, input_file_path_prompts: str) -> bool:
    sentences_number = TotalSentences(input_file_path_prompts)
    current_loop_index = 0

    while current_loop_index <= sentences_number:
        image_generation_prompt = ExtractPrompt(input_file_path_prompts)
        output_image_name = f"image_{current_loop_index}.png"
        status = generate_image(proxy_file_path, image_generation_prompt, output_image_name)

        if not status:
            return status  # False

        current_loop_index += 1

    return True

proxy_file_path_now = '../proxies.txt'
input_file_path_right_now = '../prompts.txt'
print(generate_images(proxy_file_path_now, input_file_path_right_now))